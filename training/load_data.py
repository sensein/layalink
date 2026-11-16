"""Load the datasets under data/ for training: no model dependencies, no random decision split.

    data/train/<dataset>/train.jsonl, calibration.jsonl     built by datagen/build_*.py (./build_all.sh);
                                                             <dataset> may be nested, e.g. ner_bionlp/BC2GM
    data/test/<dataset>/test.jsonl
    data/test/bioml2026/benchmark/<task>/                    Bio-ML held-out comparison (never loaded for training)

Datasets are built independently, so leakage is re-checked ACROSS them here. A train or calibration row is
dropped when it shares, with any test row (or a Bio-ML benchmark row), its exact input, its concept pair
(unordered canonical ids) or its sentence context; train rows are likewise dropped against calibration. Rows
touching a protected Bio-ML held-out query are dropped from train and calibration. The counts are returned in
the report and printed.
"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

SPLITS = ('train', 'calibration', 'test')


def _norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


def available(root):
    """Every dataset folder under data/train (nested ones such as ner_bionlp/BC2GM included)."""
    d = Path(root) / 'data' / 'train'
    return sorted(p.parent.relative_to(d).as_posix() for p in d.rglob('train.jsonl')) if d.exists() else []


def resolve(root, datasets):
    """'all' | 'a,b' | list -> dataset names; a group name (e.g. ner_bionlp) selects all of its sub-datasets."""
    avail = available(root)
    if datasets in (None, 'all', ['all']):
        return avail
    want = [d.strip() for d in datasets.split(',') if d.strip()] if isinstance(datasets, str) else list(datasets)
    out = []
    for w in want:
        hit = [a for a in avail if a == w or a.startswith(w.rstrip('/') + '/')]
        if not hit:
            raise FileNotFoundError(f'dataset {w!r} not built (available: {avail}); run ./build_all.sh')
        out += [h for h in hit if h not in out]
    return out


def _read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def row_keys(r):
    """Leakage keys: exact input, canonical concept pair, sentence context."""
    keys = {'input:' + hashlib.sha1(json.dumps({'state': r['state'], 'questions': r['questions']},
                                                sort_keys=True).encode()).hexdigest()}
    m = r.get('meta') or {}
    if r.get('task') == 'ontology_alignment' and m.get('source_curie') and m.get('target_curie'):
        keys.add('pair:' + '|'.join(sorted((m['source_curie'], m['target_curie']))))
    ctx = r['state'].get('context') if isinstance(r['state'], dict) else None
    if ctx:
        keys.add('ctx:' + _norm(ctx))
    return keys


def _touches_protected(r, ids, pairs):
    m = r.get('meta') or {}
    a, b = m.get('source_curie'), m.get('target_curie')
    if r.get('task') != 'ontology_alignment' or not (a and b):
        return False
    return a in ids or b in ids or '|'.join(sorted((a, b))) in pairs


def load_splits(root, datasets='all', return_report=False):
    root = Path(root)
    names = resolve(root, datasets)
    if not names:
        raise FileNotFoundError(f'no datasets under {root / "data" / "train"}: run ./build_all.sh first')
    rows = {s: [] for s in SPLITS}
    for n in names:
        for split in SPLITS:
            path = root / 'data' / ('test' if split == 'test' else 'train') / n / f'{split}.jsonl'
            if not path.exists():
                raise FileNotFoundError(f'{path}: run ./build_all.sh (or the matching datagen/build_*.py)')
            for r in _read(path):
                r['dataset'] = n
                r['id'] = f'{n}/{r["id"]}'
                r['group_id'] = f'{n}/{r["group_id"]}'
                rows[split].append(r)
    # Bio-ML benchmark: protected ids + labelled valid pairs count as test for leakage purposes
    prot_ids, prot_pairs, bench_keys = set(), set(), set()
    for p in sorted((root / 'data' / 'test').glob('*/benchmark/protected_ids.json')):
        d = json.loads(p.read_text())
        prot_ids |= set(d['ids'])
        prot_pairs |= set(d['pairs'])
        for vp in sorted(p.parent.glob('*/valid_pairs.jsonl')):
            for r in _read(vp):
                bench_keys |= row_keys(r)
    report = {'datasets': names, 'dropped': Counter()}
    for split in ('train', 'calibration'):
        kept = []
        for r in rows[split]:
            if _touches_protected(r, prot_ids, prot_pairs):
                report['dropped'][f'{split}:{r["dataset"]}:protected_bioml'] += 1
            else:
                kept.append(r)
        rows[split] = kept
    test_keys = set(bench_keys)
    for r in rows['test']:
        test_keys |= row_keys(r)
    cal = []
    for r in rows['calibration']:
        if row_keys(r) & test_keys:
            report['dropped'][f'calibration:{r["dataset"]}:overlaps_test'] += 1
        else:
            cal.append(r)
    rows['calibration'] = cal
    cal_keys = set()
    for r in cal:
        cal_keys |= row_keys(r)
    tr = []
    for r in rows['train']:
        k = row_keys(r)
        if k & test_keys:
            report['dropped'][f'train:{r["dataset"]}:overlaps_test'] += 1
        elif k & cal_keys:
            report['dropped'][f'train:{r["dataset"]}:overlaps_calibration'] += 1
        else:
            tr.append(r)
    rows['train'] = tr
    # final hard checks
    ids = [r['id'] for rs in rows.values() for r in rs]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate row id')
    groups = {s: {r['group_id'] for r in rs} for s, rs in rows.items()}
    for a, b in [('train', 'calibration'), ('train', 'test'), ('calibration', 'test')]:
        if groups[a] & groups[b]:
            raise ValueError(f'group leakage between {a} and {b}')
        if not rows[a] or not rows[b]:
            raise ValueError(f'empty split after leakage filtering ({a}/{b})')
    report['dropped'] = dict(sorted(report['dropped'].items()))
    report['counts'] = {s: dict(Counter(r['dataset'] for r in rs)) for s, rs in rows.items()}
    print('load_data:', json.dumps(report, indent=1), file=sys.stderr)
    return (rows, report) if return_report else rows


def preprocess(tok, rows, max_len, head_max_len):
    from laya.train import items_from_rows
    from laya.common import encode_text, serialize_state
    items = {}
    for split, rs in rows.items():
        longest = max(len(encode_text(tok, serialize_state(r['state']))['input_ids']) for r in rs)
        if longest > max_len - head_max_len - 8:
            raise ValueError(f'{split}: state has {longest} tokens; shorten metadata or increase --max-len. '
                             'Refusing silent truncation.')
        items[split], skipped = items_from_rows(tok, rs, max_len, head_max_len)
        if any(skipped.values()):
            raise ValueError(f'{split}: skipped {skipped}')
        if len(items[split]) != sum(len(r['questions']) for r in rs):
            raise ValueError('Missing decisions')
    return items
