#!/bin/bash
#SBATCH --job-name=layalink
#SBATCH --partition=pi_satra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2               # QoS pi_satra allows gpu:h100=2 (whole group) and gpu:a100=4
##SBATCH --qos=pi_satra                 # uncomment if your account does not apply this QoS by default
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue                       # allow the job to put itself back in the queue
#SBATCH --signal=B:USR1@600             # 10 min before the time limit -> requeue and resume (see trap below)
#
# Checkpointing / automatic resume:
#   training writes models/LayaLink/resume/state.pt (model + optimizer + scheduler + position) every LAYA_CKPT_EVERY
#   optimizer updates (200), every LAYA_CKPT_MINUTES (20) and at each epoch end. Any restart continues from it:
#   - time limit: the USR1 trap requeues this job, which resumes on its next start (up to MAX_REQUEUES=5 times)
#   - crash / NCCL error / OOM: torchrun restarts the workers in-job (MAX_RESTARTS=2), which resume
#   - node failure / preemption: Slurm requeues (--requeue), which resumes
#   - manual: just `sbatch train_layalink_slurm.sh` again
#   A finished model (models/LayaLink/evaluation.json, no resume state) is not retrained: FORCE_TRAIN=1 starts over.
#
# LayaLink fine-tuning on 2x H100 (DDP, bf16), then the Bio-ML 2026 comparison.
#   sbatch train_layalink_slurm.sh
#   sbatch --export=ALL,DATASETS=bioml2026,oaei_anatomy,ner_bionlp train_layalink_slurm.sh   # train on a subset
#   sbatch --export=ALL,LAYA_EPOCHS=6,LAYA_MICRO_BATCH=16,LAYA_GRAD_ACCUM=2 train_layalink_slurm.sh
#   sbatch --gres=gpu:a100:4 --export=ALL,NGPU=4,LAYA_GRAD_ACCUM=2 train_layalink_slurm.sh    # 4x A100 instead (same batch 64)
#   sbatch --export=ALL,COMPARE_BASE=1 train_layalink_slurm.sh       # evaluation also scores the untuned base model
# The group shares 2 H100s: if a labmate holds one, the job waits (squeue -p pi_satra).
#
# Upload to the cluster (data is already built; datagen/ and cache/ are NOT needed):
#   data/  training/  train_layalink_slurm.sh  requirements-training.txt      (+ evaluation/ to evaluate)
# Outputs (all saved automatically):
#   models/LayaLink/             the model (+ evaluation.json, checkpoint_latest/, resume/ while unfinished)
#   models/LayaLink/training/    training_log.jsonl (loss, LR every 50 micro-batches + per epoch),
#                                training_curves.png, training_summary.json
#   results/LayaLink-<time>/     evaluation: SUMMARY.md, per_dataset/, bioml2026/ (CodaBench TSV), baseline_fuzzy/
#   logs/layalink-<jobid>.out    the job log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

# ---- environment (set up once on a login node, see README "Running on the cluster"):
#   1. an env already active when you ran sbatch (sbatch passes your environment to the job), or
#   2. VENV=/path/to/venv, or
#   3. a venv named layalink in ./layalink, ./.venv, ~/layalink, ~/venvs/layalink or ~/.venvs/layalink, or
#   4. a conda env named layalink (CONDA_ENV to change)
# Python comes from the miniforge module. No `module load cuda`: the PyTorch wheels bundle the CUDA runtime and the
# GPU node provides the driver (the cuda/* modules are only needed to compile CUDA code).
set +u                                  # module/conda scripts reference unset variables
ACTIVE_VENV="${VIRTUAL_ENV:-}"
ACTIVE_CONDA=""; [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]] && ACTIVE_CONDA="$CONDA_DEFAULT_ENV"
module load "${MINIFORGE_MODULE:-miniforge/26.7.2-0}"
if [[ -n "$ACTIVE_VENV" ]]; then
  source "$ACTIVE_VENV/bin/activate"; echo "env: venv $ACTIVE_VENV (active at submission)"
elif [[ -n "$ACTIVE_CONDA" ]]; then
  eval "$(conda shell.bash hook)"; conda activate "$ACTIVE_CONDA"; echo "env: conda $ACTIVE_CONDA (active at submission)"
else
  FOUND="${VENV:-}"
  if [[ -z "$FOUND" ]]; then
    for d in "$PWD/layalink" "$PWD/.venv" "$HOME/layalink" "$HOME/venvs/layalink" "$HOME/.venvs/layalink"; do
      [[ -f "$d/bin/activate" ]] && { FOUND="$d"; break; }
    done
  fi
  if [[ -n "$FOUND" ]]; then
    source "$FOUND/bin/activate"; echo "env: venv $FOUND"
  else
    eval "$(conda shell.bash hook)"
    CONDA_ENV="${CONDA_ENV:-layalink}"
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
      conda activate "$CONDA_ENV"; echo "env: conda $CONDA_ENV"
    else
      echo "no Python environment found. On a login node:  module load miniforge/26.7.2-0"
      echo "  python -m venv ~/layalink && source ~/layalink/bin/activate && pip install -r requirements-training.txt"
      echo "then resubmit (or pass VENV=/path/to/venv: sbatch --export=ALL,VENV=... train_layalink_slurm.sh)"
      exit 1
    fi
  fi
fi
set -u
python -c 'import sys, torch, laya; print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda, "| GPUs", torch.cuda.device_count())'

export HF_HOME="${HF_HOME:-${SCRATCH:-$PWD}/.hf_cache}"
export USE_TF=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / 2 ))
export LAYA_AMP_DTYPE="${LAYA_AMP_DTYPE:-bf16}"
DATASETS="${DATASETS:-all}"
OUT_DIR="${OUT_DIR:-models/LayaLink}"
NGPU="${NGPU:-2}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname) | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[[ -d data/train ]] || { echo "no data/: run ./build_all.sh on a login node first"; exit 1; }

# ---- requeue 10 minutes before the time limit; the next start resumes from models/LayaLink/resume/state.pt
requeue() {
  local n="${SLURM_RESTART_COUNT:-0}"
  if [[ -n "${SLURM_JOB_ID:-}" && "$n" -lt "${MAX_REQUEUES:-5}" ]]; then
    echo "$(date): time limit near -> requeue #$((n + 1)) of job $SLURM_JOB_ID (will resume from checkpoint)"
    scontrol requeue "$SLURM_JOB_ID"
  else
    echo "$(date): time limit near, requeue limit reached; resubmit manually to resume"
  fi
}
trap requeue USR1

if [[ -f "$OUT_DIR/evaluation.json" && ! -f "$OUT_DIR/resume/state.pt" && "${FORCE_TRAIN:-0}" != "1" ]]; then
  echo "$OUT_DIR/ is already trained -> skipping training (FORCE_TRAIN=1 retrains from scratch)"
else
  [[ "${FORCE_TRAIN:-0}" == "1" ]] && rm -rf "$OUT_DIR/resume" "$OUT_DIR/evaluation.json"
  [[ -f "$OUT_DIR/resume/state.pt" ]] && echo "found $OUT_DIR/resume/state.pt -> training will RESUME"
  python training/train_layalink_mps.py --datasets "$DATASETS" --dry-run    # loads data + cross-dataset leakage check

  # base checkpoint: the revision is fixed at the first start and reused by every resume
  mkdir -p "$OUT_DIR"
  REV_FILE="$OUT_DIR/base_revision.txt"
  [[ -s "$REV_FILE" ]] || python -c 'from huggingface_hub import model_info; print(model_info("convaiinnovations/laya").sha)' > "$REV_FILE"
  MODEL_DIR=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('convaiinnovations/laya', revision='$(cat "$REV_FILE")'))")
  echo "base model: $MODEL_DIR (revision $(cat "$REV_FILE"))"

  # run in the background + wait, so the USR1 trap fires while training runs
  torchrun --standalone --nproc_per_node="$NGPU" --max-restarts="${MAX_RESTARTS:-2}" \
           training/train_layalink_ddp.py "$MODEL_DIR" "$OUT_DIR" . "$DATASETS" &
  TRAIN_PID=$!
  set +e; wait "$TRAIN_PID"; rc=$?
  while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; rc=$?; done     # wait again after a trap
  set -e
  [[ $rc -eq 0 ]] || { echo "training exited with $rc; resume state kept in $OUT_DIR/resume/"; exit "$rc"; }
fi

# ---- evaluation: every test set per dataset + Bio-ML ranking (valid metrics + CodaBench test TSV) + baseline
if [[ -f evaluation/run_evaluation.sh ]]; then
  MODEL="$OUT_DIR" DEVICE=cuda BATCH=256 DATASETS="$DATASETS" COMPARE_BASE="${COMPARE_BASE:-0}" \
    bash evaluation/run_evaluation.sh
  echo "done $(date): model in $OUT_DIR/, training curves in $OUT_DIR/training/, evaluation in results/"
else
  echo "done $(date): model in $OUT_DIR/ (evaluation/ not uploaded -> evaluation skipped)"
fi
