import json, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent   # repository root
sys.path.insert(0, str(ROOT / 'training'))
import plot_training  # noqa: E402


def write_log(model_dir, records):
    d = Path(model_dir, 'training')
    d.mkdir(parents=True)
    (d / 'training_log.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))


class PlotTests(unittest.TestCase):
    def test_ddp_log_with_resume(self):
        recs = [{'event': 'start', 'epoch': 0, 'update': 0}]
        for ep in (1, 2):
            for mb in range(50, 501, 50):
                recs.append({'event': 'step', 'epoch': ep, 'micro_batch': mb, 'update': (ep - 1) * 125 + mb // 4,
                             'loss': 1.0 / (ep + mb / 500), 'lr_encoder': 2e-5, 'lr_head': 1e-4, 'sigma': 0.4})
            recs.append({'event': 'epoch_end', 'epoch': ep, 'avg_loss': 0.8 / ep})
        # a resume replays the last steps of epoch 2: the duplicates must be kept once
        recs.append({'event': 'resume', 'epoch': 1, 'update': 200})
        recs += [r for r in recs if r.get('event') == 'step' and r['epoch'] == 2 and r['micro_batch'] > 400]
        with tempfile.TemporaryDirectory() as d:
            write_log(d, recs)
            out = plot_training.plot(d)
            self.assertTrue(out.exists() and out.stat().st_size > 10_000)

    def test_epoch_only_log(self):                     # MPS/CPU script and notebook
        with tempfile.TemporaryDirectory() as d:
            write_log(d, [{'event': 'epoch_end', 'epoch': e, 'avg_loss': 1 / e} for e in (1, 2, 3)])
            self.assertTrue(plot_training.plot(d).exists())


if __name__ == '__main__':
    unittest.main()
