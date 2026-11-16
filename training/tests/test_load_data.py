import json, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent   # repository root
sys.path.insert(0, str(ROOT / 'training'))
import load_data as training_data  # noqa: E402

Q = {'relation': {'type': 'choice', 'instructions': 'x', 'criteria': {'equivalent': 'e', 'no_match': 'n'}}}


def row(i, grp, a, b, label='equivalent'):
    return {'id': i, 'group_id': grp, 'task': 'ontology_alignment', 'state': {'source': {'label': a}, 'target': {'label': b}},
            'questions': Q, 'gold': {'relation': {'label': label, 'probabilities': {'equivalent': 1.0, 'no_match': 0.0}}},
            'meta': {'source_curie': a, 'target_curie': b}}


def write(root, split_dir, name, split, rows):
    d = Path(root, 'data', split_dir, name)
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


class CrossDatasetLeakage(unittest.TestCase):
    def test_overlap_and_protected_rows_are_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            # dataset A test contains pair x|y; dataset B trains on y|x (reversed) -> must be dropped
            write(root, 'train', 'A', 'train', [row('a1', 'g1', 'p:1', 'q:1')])
            write(root, 'train', 'A', 'calibration', [row('a2', 'g2', 'p:2', 'q:2')])
            write(root, 'test', 'A', 'test', [row('a3', 'g3', 'x:1', 'y:1')])
            write(root, 'train', 'B', 'train', [row('b1', 'h1', 'y:1', 'x:1'), row('b2', 'h2', 'ncit:C9', 'doid:5'),
                                               row('b3', 'h3', 'p:7', 'q:7')])
            write(root, 'train', 'B', 'calibration', [row('b4', 'h4', 'p:8', 'q:8')])
            write(root, 'test', 'B', 'test', [row('b5', 'h5', 'p:9', 'q:9')])
            bench = Path(root, 'data', 'test', 'B', 'benchmark')
            bench.mkdir(parents=True)
            (bench / 'protected_ids.json').write_text(json.dumps({'ids': ['ncit:C9'], 'pairs': []}))
            rows, rep = training_data.load_splits(root, 'all', return_report=True)
            self.assertEqual(sorted(r['id'] for r in rows['train']), ['A/a1', 'B/b3'])
            self.assertEqual(rep['dropped'], {'train:B:overlaps_test': 1, 'train:B:protected_bioml': 1})


class NestedDatasets(unittest.TestCase):
    def test_nested_discovery_and_group_selection(self):
        with tempfile.TemporaryDirectory() as root:
            for i, name in enumerate(['ner_bionlp/BC2GM', 'ner_bionlp/s800', 'oaei_anatomy']):
                write(root, 'train', name, 'train', [row(f't{i}', f'g{i}', f'a:{i}', f'b:{i}')])
                write(root, 'train', name, 'calibration', [row(f'c{i}', f'h{i}', f'c:{i}', f'd:{i}')])
                write(root, 'test', name, 'test', [row(f'x{i}', f'k{i}', f'e:{i}', f'f:{i}')])
            self.assertEqual(training_data.available(root), ['ner_bionlp/BC2GM', 'ner_bionlp/s800', 'oaei_anatomy'])
            self.assertEqual(training_data.resolve(root, 'ner_bionlp'), ['ner_bionlp/BC2GM', 'ner_bionlp/s800'])
            rows = training_data.load_splits(root, 'ner_bionlp/s800,oaei_anatomy')
            self.assertEqual(sorted(r['dataset'] for r in rows['train']), ['ner_bionlp/s800', 'oaei_anatomy'])


if __name__ == '__main__':
    unittest.main()
