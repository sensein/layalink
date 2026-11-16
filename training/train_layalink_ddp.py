"""Multi-GPU DDP fine-tuning of LayaLink (2xT4 on Kaggle, 2xH100 via train_layalink_slurm.sh), adapted from the Laya upstream notebook.
Source: https://github.com/NandhaKishorM/laya/blob/80a1da1fa5448ea12163f4d977935625a0aea670/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
Apache-2.0; upstream license supplied in UPSTREAM_LICENSE.
Changes: local ontology JSONL, separate component splits, local evaluation, no publishing.

Training semantics -- item construction, encoding/collation, the RLCD objective,
the sigma schedule, the calibration holdout and the temperature fit -- come from
the canonical primitives in `laya.train` / `laya.common` / `laya.calibrate`.
What intentionally stays notebook-private is the distributed orchestration:
process-group setup, rank/local-rank wiring, the DDP wrapper, equal-rank
sharding and the effective-batch accounting.
"""
import os, sys, time, json, random, math
from datetime import timedelta
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.common import build_model, collate_items
from laya.calibrate import fit_temperature_map
from laya.train import (
    calibration_records,
    encode_item,
    rlcd_loss,
    save_checkpoint,
    sigma_at,
    split_calibration,
)

# ---------------------------------------------------------------- resumable checkpoints
# OUTPUT_DIR/resume/state.pt holds model + optimizer + scheduler + scaler + position; it is written atomically by
# rank 0 at optimizer-update boundaries (every LAYA_CKPT_EVERY updates, every LAYA_CKPT_MINUTES, and at each
# epoch end) and removed once the final calibrated model is saved. A restart (Slurm requeue, torchrun
# --max-restarts, or a manual re-submit) continues from it when the run fingerprint matches.
# LAYA_RESUME=auto (default) | never.
def _fingerprint(**kw):
    return json.dumps(kw, sort_keys=True)


def _save_state(path, model, optimizer, scheduler, scaler, fingerprint, epoch, micro_done, updates,
                epoch_loss, n_batches):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "fingerprint": fingerprint, "epoch": epoch, "micro_done": micro_done, "updates": updates,
                "epoch_loss": epoch_loss, "n_batches": n_batches, "saved_at": time.time()}, tmp)
    os.replace(tmp, path)


def _log(output_dir, **rec):
    """Append one record to OUTPUT_DIR/training/training_log.jsonl (rank 0 only; survives resumes)."""
    d = os.path.join(output_dir, "training")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "training_log.jsonl"), "a") as f:
        f.write(json.dumps(dict(rec, time=round(time.time(), 1))) + "\n")


def main():
    dist.init_process_group("nccl", timeout=timedelta(hours=4))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if len(sys.argv) != 5:
        raise ValueError("Usage: torchrun --standalone --nproc_per_node=2 training/train_layalink_ddp.py MODEL_DIR OUTPUT_DIR DATA_ROOT DATASETS(all|a,b)")
    model_dir, output_dir, data_root, datasets = sys.argv[1:]
    from load_data import load_splits, preprocess
    rows = load_splits(data_root, datasets)

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256
    max_len = cfg["max_len"]
    head_max_len = cfg["head_max_len"]

    from laya.agent import _fix_tokenizer_config
    if rank == 0:
        _fix_tokenizer_config(model_dir)
    dist.barrier()
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    torch.manual_seed(42)
    np.random.seed(42)
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))

    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    split_items = preprocess(tok, rows, max_len, head_max_len)
    train_items = split_items["train"]
    calib_items = split_items["calibration"]
    test_items = split_items["test"]

    # Trim before sharding so every rank gets the same count (drops at most
    # world_size-1 items). Otherwise ranks can run different numbers of
    # micro-batches and DDP deadlocks on the unmatched all-reduce (#678).
    train_items = train_items[:len(train_items) // world_size * world_size]
    my_items = train_items[rank::world_size]

    EPOCHS = int(os.environ.get("LAYA_EPOCHS", "4"))
    MICRO_BATCH = int(os.environ.get("LAYA_MICRO_BATCH", "8"))      # 8 sequences per forward pass per GPU
    GRAD_ACCUM = int(os.environ.get("LAYA_GRAD_ACCUM", "4"))       # Effective batch across 2 GPUs = 64 sequences (8 * 2 * 4)
    GROUP_SIZE = 4       # RLCD baseline samples
    LR_ENCODER = 2.5e-5  # Encoder adaptation rate
    LR_HEAD = 1.0e-4     # Head adaptation rate
    SIGMA_START = 0.4    # Exploration noise
    SIGMA_END = 0.1

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]

    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD}
    ], weight_decay=0.01)

    total_updates = math.ceil(math.ceil(len(my_items) / MICRO_BATCH) / GRAD_ACCUM) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    # bf16 on GPUs that support it (H100/A100: no loss scaling needed), fp16 + GradScaler otherwise (T4).
    # Override with LAYA_AMP_DTYPE=bf16|fp16.
    amp_name = os.environ.get("LAYA_AMP_DTYPE") or ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_name]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    CKPT_EVERY = int(os.environ.get("LAYA_CKPT_EVERY", "200"))       # optimizer updates between resume checkpoints
    CKPT_MINUTES = float(os.environ.get("LAYA_CKPT_MINUTES", "20"))  # ... or this many minutes, whichever first
    resume_path = os.path.join(output_dir, "resume", "state.pt")
    fingerprint = _fingerprint(datasets=datasets, epochs=EPOCHS, micro_batch=MICRO_BATCH, grad_accum=GRAD_ACCUM,
                               world_size=world_size, train_items=len(train_items), calib_items=len(calib_items),
                               base_model=os.path.basename(os.path.normpath(model_dir)), amp=amp_name)
    start_epoch, start_micro, updates = 0, 0, 0
    resumed = None
    if os.environ.get("LAYA_RESUME", "auto") != "never" and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise SystemExit(f"{resume_path} was written by a different run configuration:\n  saved:   "
                             f"{state['fingerprint']}\n  current: {fingerprint}\nUse the same settings, another "
                             "OUTPUT_DIR, or LAYA_RESUME=never to start over.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, start_micro, updates = state["epoch"], state["micro_done"], state["updates"]
        resumed = state
        # RLCD exploration noise: re-seeded from the position (not bit-identical to an uninterrupted run)
        torch.manual_seed(1000003 * (updates + 1) + rank)
        if rank == 0:
            print(f"RESUMING from {resume_path}: epoch {start_epoch + 1}, micro-batch {start_micro}, "
                  f"{updates} optimizer updates done")
        del state
    dist.barrier()
    last_save = time.time()

    if rank == 0:
        _log(output_dir, event="resume" if resumed is not None else "start", epoch=start_epoch, update=updates,
             train_items=len(train_items), calib_items=len(calib_items), test_items=len(test_items),
             epochs=EPOCHS, world_size=world_size, effective_batch=MICRO_BATCH * world_size * GRAD_ACCUM,
             gpu=torch.cuda.get_device_name(local_rank), amp=amp_name, datasets=datasets)
        print(f"Starting {world_size}-GPU DDP training ({torch.cuda.get_device_name(local_rank)}, {amp_name}): {len(train_items)} train items ({len(calib_items)} held out for calibration) | {len(my_items)} per rank | {EPOCHS} epochs")
    t0 = time.time()

    base_items = list(my_items)
    for epoch in range(start_epoch, EPOCHS):
        # each epoch's order depends only on (epoch, rank), so a resumed epoch sees the same order
        my_items = list(base_items)
        random.Random(42 + 1000 * epoch + rank).shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        skip = start_micro if epoch == start_epoch else 0
        if skip and resumed is not None:
            epoch_loss, n_batches = resumed["epoch_loss"], resumed["n_batches"]

        sigma = sigma_at(epoch, EPOCHS, SIGMA_START, SIGMA_END)
        steps_per_epoch = math.ceil(len(my_items) / MICRO_BATCH)

        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            if accum_step < skip:                 # already trained before the restart
                accum_step += 1
                continue

            encoded = [encode_item(tok, it, max_len, head_max_len) for it in chunk]
            batch = collate_items([encoded], tok.pad_token_id)

            with torch.autocast("cuda", dtype=amp_dtype):
                logits, act = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device)
                )

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)

            # Canonical objective: RLCD policy gradient over noisy logits + full
            # soft cross-entropy guidance (`laya.train.rlcd_loss`).
            loss = rlcd_loss(logits, target, mask, batch["qtype"].to(device), sigma,
                             samples=GROUP_SIZE, w_sph=0.75, w_rps=1.0)

            window_start = (accum_step // GRAD_ACCUM) * GRAD_ACCUM
            window_size = min(GRAD_ACCUM, steps_per_epoch - window_start)
            scaled = loss / window_size
            scaler.scale(scaled).backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                # resume checkpoint at this update boundary (rank 0 only; other ranks wait in the next all-reduce)
                if rank == 0 and (updates % CKPT_EVERY == 0 or time.time() - last_save > 60 * CKPT_MINUTES):
                    _save_state(resume_path, model, optimizer, scheduler, scaler, fingerprint, epoch, accum_step,
                                updates, epoch_loss + loss.item(), n_batches + 1)
                    last_save = time.time()
                    print(f"  saved resume checkpoint: epoch {epoch+1}, micro-batch {accum_step}, update {updates}")

            epoch_loss += loss.item()
            n_batches += 1

            if rank == 0 and (n_batches % 50) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {n_batches} | Loss: {loss.item():.4f} | LR: {cur_lr:.2e}")
                _log(output_dir, event="step", epoch=epoch + 1, micro_batch=n_batches, update=updates,
                     loss=round(loss.item(), 5), running_loss=round(epoch_loss / max(1, n_batches), 5),
                     lr_encoder=cur_lr, lr_head=scheduler.get_last_lr()[-1], sigma=sigma)

        if rank == 0:
            print(f"=== Epoch {epoch+1}/{EPOCHS} Completed in {time.time()-t0:.1f}s | Avg Loss: {epoch_loss/max(1, n_batches):.4f} ===")
            _log(output_dir, event="epoch_end", epoch=epoch + 1, update=updates,
                 avg_loss=round(epoch_loss / max(1, n_batches), 5), seconds=round(time.time() - t0, 1))

        dist.barrier()

        # Overwrite a single rolling checkpoint after each epoch so a crash,
        # OOM, or Kaggle session timeout doesn't lose all prior training.
        if rank == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoint_latest")
            save_checkpoint(model, tok, dict(cfg, max_len=max_len, head_max_len=head_max_len), ckpt_dir)
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "total_epochs": EPOCHS,
                    "avg_loss": epoch_loss / max(1, n_batches)
                }, f, indent=2)
            print(f"  Saved rolling checkpoint (epoch {epoch+1}/{EPOCHS}) to {ckpt_dir}")
            _save_state(resume_path, model, optimizer, scheduler, scaler, fingerprint, epoch + 1, 0, updates, 0.0, 0)
            last_save = time.time()
        dist.barrier()

    dist.barrier()

    # Post-training temperature calibration on rank 0
    if rank == 0:
        print("\nFitting post-training calibration temperatures...")
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        records = calibration_records(model, tok, calib_items, device, max_len, head_max_len, batch_size=2)
        fitted = fit_temperature_map(records)
        fitted_temps = fitted["temperature"]
        print("Fitted calibration temperatures:", fitted_temps)

        out_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len)
        out_cfg["fine_tuned"] = True
        out_cfg["model_name"] = "LayaLink"
        if fitted_temps is not None:
            out_cfg["temperature"] = fitted_temps
            # This fit carries per-bucket overrides; inherited ones would shadow them.
            out_cfg.pop("temperature_by_options", None)
            if fitted["temperature_by_options"]:
                out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
        save_checkpoint(model, tok, out_cfg, output_dir)
        from laya.train import evaluate_records
        test_records = calibration_records(model, tok, test_items, device, max_len, head_max_len, batch_size=2)
        report = {"datasets": datasets, "official_oaei_evaluation": False,
                  "train_decisions_used": len(train_items), "world_size": world_size,
                  "raw": evaluate_records(test_records),
                  "calibrated": evaluate_records(test_records, fitted_temps, fitted["temperature_by_options"])}
        with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
            json.dump(report, f, indent=2, default=float)
        _log(output_dir, event="done", update=updates, temperature=fitted_temps,
             test_accuracy_raw=report["raw"].get("accuracy"), test_accuracy_calibrated=report["calibrated"].get("accuracy"))
        with open(os.path.join(output_dir, "training", "training_summary.json"), "w") as f:
            json.dump({"datasets": datasets, "epochs": EPOCHS, "updates": updates, "world_size": world_size,
                       "effective_batch": MICRO_BATCH * world_size * GRAD_ACCUM, "amp": amp_name,
                       "lr_encoder": LR_ENCODER, "lr_head": LR_HEAD, "temperature": fitted_temps,
                       "test": {"raw": report["raw"], "calibrated": report["calibrated"]}}, f, indent=2, default=float)
        try:                                       # loss / LR curves; never fails the run
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from plot_training import plot
            print("training curves:", plot(output_dir))
        except Exception as e:
            print(f"(plotting skipped: {type(e).__name__}: {e})")
        print(json.dumps(report, indent=2, default=float))
        if os.path.exists(resume_path):
            os.remove(resume_path)                 # training finished: nothing left to resume
        print(f"Model successfully saved to {output_dir}!")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
