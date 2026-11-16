"""Publish the fine-tuned LayaLink model to the Hugging Face Hub as <your-namespace>/layalink.

    hf auth login                                      # once (write token)
    python training/publish_hf.py --dry-run            # list what would be uploaded, write the model card
    python training/publish_hf.py                      # upload (repo is created PRIVATE)
    python training/publish_hf.py --repo my-org/layalink --public

Uploads the final calibrated model folder (default models/LayaLink/) plus a generated README.md model card,
including training/ (loss log, curves, summary). Not uploaded: checkpoint_latest/, resume/ (optimizer state),
predictions.json and macOS .DS_Store files. Nothing is uploaded without running this script yourself.

The model card is built from what is on disk:
  models/LayaLink/training/training_summary.json, evaluation.json   training settings + pooled test metrics
  results/<model>-<time>/   (newest full run of evaluation/run_evaluation.sh)
      per_dataset/metrics.json      per dataset x question accuracy / macro-F1 / ECE
      per_dataset/predictions.jsonl the model's answers, shown for the worked examples
      bioml2026/, baseline_fuzzy/   Bio-ML 2026 ranking (LayaLink vs fuzzy baseline)
      base_model/                   untuned convaiinnovations/laya on the same test sets (COMPARE_BASE=1)
  data/test/                        one real test row per task for the worked examples
"""
import argparse, json, pprint, re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent      # repository root
REPO_NAME = 'layalink'
EXCLUDE = ['checkpoint_latest/*', 'resume/*', 'predictions.json', '*.part', '*.tmp', '.DS_Store', '**/.DS_Store']

# one worked example per task: (task, preferred dataset prefix, question, gold label to show; None = any but 'none')
EXAMPLES = [
    ('ontology_alignment', 'oaei_anatomy', 'relation', 'equivalent'),
    ('ontology_alignment', 'bioml2026', 'relation', 'source_narrower'),
    ('entity_linking', 'ncbi_corpora', 'mapping', 'match'),
    ('concept_selection', 'ner_bionlp/', 'concept', None),
]


def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def _laya_pin():
    req = HERE / 'requirements-training.txt'
    for line in req.read_text().splitlines() if req.exists() else []:
        if line.strip().startswith('laya'):
            return line.split('#')[0].strip()
    return 'laya'


def latest_results(model_name):
    """Newest results/<model>-<time>/ from a full (unlimited) evaluation run, or None."""
    runs = []
    for m in (HERE / 'results').glob(f'{model_name}-*/per_dataset/metrics.json'):
        d = _read_json(m)
        if d and not d.get('limit'):
            runs.append(m.parent.parent)
    return max(runs, key=lambda p: p.name) if runs else None


def _predictions(run):
    out = {}
    p = run / 'per_dataset' / 'predictions.jsonl' if run else None
    if p and p.exists():
        for line in p.open(encoding='utf-8'):
            r = json.loads(line)
            out[(r['dataset'], r['id'])] = r
    return out


def pick_examples(preds):
    """First test row (deterministic file/line order) per EXAMPLES entry that the model answered correctly."""
    test = HERE / 'data' / 'test'
    files = sorted(test.rglob('test.jsonl')) if test.exists() else []
    picked = []
    for task, prefix, qid, want in EXAMPLES:
        for f in files:
            ds = f.parent.relative_to(test).as_posix()
            if not ds.startswith(prefix):
                continue
            hit = None
            for line in f.open(encoding='utf-8'):
                r = json.loads(line)
                g = r['gold'].get(qid, {}).get('label')
                if r['task'] != task or g is None or (want and g != want) or (not want and g == 'none'):
                    continue
                if len(json.dumps(r['state'])) > 1500:              # keep the card readable
                    continue
                ctx = r['state'].get('context', '')
                if ctx and (len(ctx) < 60 or ctx.endswith('...')):     # want a full sentence, not a trimmed one
                    continue
                p = preds.get((ds, r['id']))
                if preds and not (p and all(p['pred'][q]['label'] == p['gold'][q] for q in r['questions'])):
                    continue
                if p and max(p['pred'][qid]['probabilities'].values()) < 0.6:   # a clear answer makes a better example
                    continue
                hit = (ds, r, p)
                break
            if hit:
                picked.append(hit)
                break
    return picked


def _py(obj, indent):
    """Python literal (valid code, unlike JSON's true/false/null), indented to sit after `name = `."""
    s = pprint.pformat(obj, width=108 - indent, sort_dicts=False)
    return s.replace('\n', '\n' + ' ' * indent)


def example_md(ds, r, p):
    q0 = next(iter(r['questions']))
    head = {'ontology_alignment': 'Ontology alignment: how does SOURCE relate to TARGET?',
            'entity_linking': 'Entity linking: does this candidate match the mention in context?',
            'concept_selection': 'Concept selection: which candidate (or none) matches the mention?'}[r['task']]
    head += f" (`{r['gold'][q0]['label']}`)" if r['task'] == 'ontology_alignment' else ''
    lines = [f"### {head}", '',
             f"A real row from the held-out `{ds}` test set (provenance: `{r.get('provenance', '?')}`).", '',
             '```python', f"state = {_py(r['state'], 8)}", '', f"questions = {_py(r['questions'], 12)}", '',
             'result = agent.predict(state, questions)', '```', '']
    if p:
        out = {}
        for q, a in p['pred'].items():
            out[q] = {'label': a['label'], 'probabilities': dict(sorted(a['probabilities'].items(), key=lambda kv: -kv[1]))}
        lines += ["LayaLink's answer (from the evaluation run), with the gold label:", '', '```python']
        for q, a in out.items():
            probs = ', '.join(f"'{k}': {v:.4f}" for k, v in a['probabilities'].items())
            lines.append(f"result['answers']['{q}']  # -> {a['label']!r}  (gold: {p['gold'][q]!r})  {{{probs}}}")
        lines += ['```', '']
    return '\n'.join(lines)


def _f(x, nd=3):
    return '' if x is None else f'{x:.{nd}f}'


def results_md(run, ev, model_name):
    parts = []
    pooled = [(k, ev[k]) for k in ('raw', 'calibrated') if isinstance(ev.get(k), dict)]
    if pooled:
        parts += ['### All test sets pooled', '',
                  'Measured right after training on every `data/test/*` set together (each question counted once). '
                  '"Calibrated" applies the temperatures fitted on the calibration split.', '',
                  '| Probabilities | Decisions | Accuracy | ECE | Brier | Mean confidence |', '|---|---|---|---|---|---|']
        parts += [f"| {k} | {t['items']} | {_f(t['accuracy'], 4)} | {_f(t['ece'], 4)} | {_f(t['brier'], 4)} | "
                  f"{_f(t['mean_confidence'], 4)} |" for k, t in pooled]
        parts.append('')
    if run:
        m = _read_json(run / 'per_dataset' / 'metrics.json').get('metrics', {})
        base = _read_json(run / 'base_model' / 'metrics.json').get('metrics', {})
        if m:
            parts += ['### Per test set', '',
                      f"From `evaluation/run_evaluation.sh` (`{run.relative_to(HERE).as_posix()}`). Accuracy and macro-F1 "
                      'over the labels of each question; ECE = expected calibration error (lower is better). '
                      '`ner_bionlp` labels are silver (lexical/semantic evidence), the others gold.'
                      + (' "Base" is the untuned `convaiinnovations/laya` on the same rows.' if base else ''), '',
                      '| Test set | Task / question | n | Accuracy | Macro-F1 | ECE |' + (' Base accuracy |' if base else ''),
                      '|---|---|---|---|---|---|' + ('---|' if base else '')]
            for ds, qs in m.items():
                for q, s in qs.items():
                    if not isinstance(s, dict):                   # e.g. '_seconds'
                        continue
                    row = f"| {ds} | {q} | {s['n']} | {_f(s['accuracy'])} | {_f(s['macro_f1'])} | {_f(s['ece'], 4)} |"
                    if base:
                        row += f" {_f(base.get(ds, {}).get(q, {}).get('accuracy'))} |"
                    parts.append(row)
            parts.append('')
        ranking = []
        for label, pattern in ((model_name, f'bioml2026/*.{model_name}.report.json'),
                               ('fuzzy label baseline', 'baseline_fuzzy/*.fuzzy.report.json')):
            for p in sorted(run.glob(pattern)):
                r = _read_json(p)
                if 'valid' in r:
                    v = r['valid']
                    ranking.append(f"| {label} | {r['task']} | {v['queries']} | {v['MRR']} | {v['Hits@1']} | {v['Hits@5']} | {v['Hits@10']} |")
        if ranking:
            parts += ['### Bio-ML 2026 local ranking (OAEI)', '',
                      'Rank the 100 released candidates per source concept; public **valid** split, scored exactly like the '
                      "track's `scoring_kit/score_local.py`. The test gold is hidden: official test numbers come only "
                      'from the organisers (CodaBench).', '',
                      '| System | Pair | Queries | MRR | Hits@1 | Hits@5 | Hits@10 |', '|---|---|---|---|---|---|---|', *ranking, '']
    comp = _read_json(run / 'compliance' / 'bioml2026_compliance.json') if run else {}
    comp = comp or _read_json(HERE / 'results' / 'compliance' / 'bioml2026_compliance.json')
    if comp:
        mark = {True: 'pass', False: '**FAIL**', None: 'skipped'}
        parts += ['### OAEI Bio-ML 2026 rule check', '',
                  f"**{'Compliant' if comp['compliant'] else 'NOT compliant'}**: `evaluation/check_bioml_compliance.py` "
                  f"scanned all {sum(comp['rows_scanned'].values())} rows (train, calibration, test, Bio-ML benchmark) "
                  'and the recorded build inputs. No MONDO or UMLS identifier, IRI, ontology or source is used, and no '
                  'Bio-ML valid/test query reaches training.', '',
                  '| Rule | Result |', '|---|---|', *[f'| {k} | {mark[v]} |' for k, v in comp['rules'].items()], '']
    if not parts:
        parts = ['Run `evaluation/run_evaluation.sh` (without `LIMIT`) before publishing to add the results here.', '']
    parts.append('The test sets are this project\'s own held-out splits (grouped so no concept, pair or sentence is shared '
                 'with training), not official benchmarks, except the Bio-ML ranking above.')
    return '\n'.join(parts)


def model_card(model_dir: Path, repo_id: str) -> str:
    ev = _read_json(model_dir / 'evaluation.json')
    summ = _read_json(model_dir / 'training' / 'training_summary.json')
    run = latest_results(model_dir.name)
    examples = pick_examples(_predictions(run))
    datasets = ev.get('datasets') or summ.get('datasets') or 'see the training repository'
    if datasets == 'all' and run:                       # name the datasets instead of 'all'
        names = [d for d in _read_json(run / 'per_dataset' / 'metrics.json').get('metrics', {}) if 'benchmark' not in d]
        datasets = names or datasets
    if isinstance(datasets, list):
        datasets = ', '.join(f'`{d}`' for d in datasets)
    train_line = ''
    if summ:
        train_line = (f"{summ.get('epochs')} epochs, {summ.get('updates')} optimizer updates, effective batch "
                      f"{summ.get('effective_batch')}, {summ.get('world_size')} GPU(s), {summ.get('amp')}; learning rates "
                      f"{summ.get('lr_encoder')} (encoder) / {summ.get('lr_head')} (head).")
    has_curves = (model_dir / 'training' / 'training_curves.png').exists()
    ex_md = '\n'.join(example_md(*e) for e in examples) or '(no test data found when the card was generated)\n'
    return f"""---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: laya
tags: [ontology-matching, ontology-alignment, entity-linking, biomedical, laya, OAEI]
---

# LayaLink

LayaLink is [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) fine-tuned with RLCD for
three typed decisions over biomedical concepts. Every answer is a calibrated probability distribution.

| Task | Input (`state`) | Questions and answers |
|---|---|---|
| **ontology_alignment** | a source and a target concept (label, definition, parents, ontology) | `relation`: `equivalent`, `source_narrower` (source is a subclass of target), `source_broader`, `related`, `no_match`, `insufficient`; `exact_match`: P(same meaning and scope) |
| **entity_linking** | a text mention, its sentence, one candidate concept | `mapping`: `match`, `no_match`, `insufficient`; `accept`: P(enough evidence to accept) |
| **concept_selection** | a mention, its sentence, a list of candidate concepts | `concept`: the matching candidate's id or `none`; `resolvable`: P(some candidate matches) |

LayaLink judges candidates you supply; it does not search an ontology. Pair it with a retriever (BioPortal, OLS,
lexical or embedding search) and let LayaLink decide among the candidates.

## Installation

```bash
pip install "{_laya_pin()}"
```

This is the `laya` version LayaLink was trained and tested with (Python 3.10+). While the repository is private, log
in first with `hf auth login` (a read token is enough).

## Quickstart

```python
import laya

agent = laya.Agent("{repo_id}", device="cuda")      # or "mps" (Apple Silicon) / "cpu"
# a local copy works the same way: laya.Agent("path/to/LayaLink")
```

If loading hangs and TensorFlow is installed, set `USE_TF=0` (as for the base model).

Ask the questions **with exactly the wording below**: it is the wording LayaLink was trained on. For
`concept_selection`, build `criteria` from your candidates (`id: "label (ONTOLOGY)"`, plus `none`).

{ex_md}
### Many inputs at once

```python
results = agent.predict_batch([state_1, state_2, state_3], questions, batch_size=64)   # one shared question set
for r in results:
    print(r["answers"])
```

Each answer has `choice` + `probabilities` (choice questions) or `noul` = P(yes) (yes/no questions), plus `confidence`.

## Results

{results_md(run, ev, model_dir.name)}

## Training

- **Data:** {datasets}. Built from OAEI 2025 Anatomy and Conference, OAEI Bio-ML 2026 (NCIT–DOID train references only),
  the NCBI normalisation corpora (NCBI-disease, BC5CDR), Biomappings, and BioNLP-Corpus NER mentions (BC2GM, BC4CHEMD,
  BC5CDR, NCBI-disease, JNLPBA, s800, linnaeus) with BioPortal/OLS4 candidates. The BioNLP-Corpus labels are silver
  (exact / fuzzy / all-mpnet-base-v2 semantic evidence). Splits are grouped (80/10/10), cross-dataset leakage was
  filtered, and Bio-ML valid/test queries were excluded. In line with the Bio-ML 2026 rules, MONDO and UMLS were not used.
- **Run:** {train_line or 'see training/training_summary.json'}
- **Logs:** `training/training_log.jsonl`, `training/training_summary.json`{', `training/training_curves.png`' if has_curves else ''} in this repository.

## Limitations

- A probability is a model estimate, not proof of logical equivalence. Check the logical consequences before asserting
  `owl:equivalentClass`.
- Subclass direction (`source_narrower` vs `source_broader`) is the hardest relation; check it on new ontology pairs.
- Some labels are derived from ontology axioms or are silver. Calibration temperatures were fitted on mixed-provenance data.
- English biomedical text and ontologies only.

Training data remain subject to their providers' licences (OAEI, Bio-ML 2026: DOID CC0 / NCIT CC BY 4.0,
Biomappings CC0, NCBI corpora, BioNLP-Corpus sources).
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, default=HERE / 'models' / 'LayaLink', help='fine-tuned model folder')
    p.add_argument('--repo', default=REPO_NAME, help="repo id or bare name (default 'layalink' under your account)")
    p.add_argument('--public', action='store_true', help='create the repo public (default: private)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if not ((args.model / 'rl_agent_config.json').exists() and (args.model / 'model.safetensors').exists()):
        raise SystemExit(f'{args.model}: no Laya checkpoint found (train first; the scripts save to models/LayaLink/)')
    from huggingface_hub import HfApi
    api = HfApi()
    if '/' in args.repo:
        repo_id = args.repo
    else:
        try:
            repo_id = f"{api.whoami()['name']}/{args.repo}"
        except Exception:
            if not args.dry_run:
                raise SystemExit('not logged in to Hugging Face: run `hf auth login` (write token) or pass --repo owner/name')
            repo_id = f'<your-account>/{args.repo}'
    card = args.model / 'README.md'
    card.write_text(model_card(args.model, repo_id))
    files = sorted(str(f.relative_to(args.model)) for f in args.model.rglob('*')
                   if f.is_file() and not f.relative_to(args.model).as_posix().startswith(('checkpoint_latest/', 'resume/'))
                   and f.name not in ('predictions.json', '.DS_Store') and not re.search(r'\.(part|tmp)$', f.name))
    print(f'repo: {repo_id} ({"public" if args.public else "private"})')
    print('files:', *files, sep='\n  ')
    if args.dry_run:
        print('dry run: nothing uploaded; model card written to', card)
        return
    api.create_repo(repo_id, private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=str(args.model), repo_id=repo_id, ignore_patterns=EXCLUDE,
                      commit_message='Upload LayaLink')
    print(f'uploaded: https://huggingface.co/{repo_id}')


if __name__ == '__main__':
    main()
