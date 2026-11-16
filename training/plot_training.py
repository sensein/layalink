"""Plot training curves from <model>/training/training_log.jsonl -> <model>/training/training_curves.png.

Runs automatically at the end of training; run it yourself any time, even while training is still going:
    python training/plot_training.py models/LayaLink
Panels: step loss (raw + moving average, epoch boundaries and resumes marked), per-epoch average loss, learning
rate (encoder and head) with the RLCD exploration noise sigma. After a resume, the repeated steps are kept once
(the last copy).
"""
import json, sys
from pathlib import Path


def read_log(model_dir):
    path = Path(model_dir) / 'training' / 'training_log.jsonl'
    if not path.exists():
        raise FileNotFoundError(f'{path}: no training log yet')
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def plot(model_dir, out=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    log = read_log(model_dir)
    steps = {}                                           # (epoch, micro_batch) -> last record (dedupe after resume)
    for r in log:
        if r.get('event') == 'step':
            steps[(r['epoch'], r['micro_batch'])] = r
    steps = sorted(steps.values(), key=lambda r: (r['epoch'], r['micro_batch']))
    epochs = {r['epoch']: r for r in log if r.get('event') == 'epoch_end'}
    resumes = [r for r in log if r.get('event') == 'resume']
    if not steps and not epochs:
        raise ValueError('no step or epoch records in the training log yet')
    out = Path(out or Path(model_dir) / 'training' / 'training_curves.png')
    if not steps:                                        # MPS/CPU script: per-epoch records only
        fig, ax = plt.subplots(figsize=(6, 4.4))
        e = sorted(epochs)
        ax.plot(e, [epochs[i]['avg_loss'] for i in e], marker='o', color='#1f4e8c')
        ax.set(title=f'{Path(model_dir).name}: average loss per epoch', xlabel='epoch', ylabel='loss', xticks=e)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return out
    x = list(range(len(steps)))
    loss = [r['loss'] for r in steps]
    k = max(1, len(loss) // 25)
    ma = [sum(loss[max(0, i - k + 1):i + 1]) / len(loss[max(0, i - k + 1):i + 1]) for i in range(len(loss))]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(x, loss, color='#9db4d0', lw=0.8, label='loss (every 50 micro-batches)')
    ax[0].plot(x, ma, color='#1f4e8c', lw=2, label=f'moving average ({k})')
    for i in range(1, len(steps)):
        if steps[i]['epoch'] != steps[i - 1]['epoch']:
            ax[0].axvline(i, color='#bbbbbb', ls='--', lw=0.8)
    for r in resumes:
        idx = next((i for i, s in enumerate(steps) if s.get('update', 0) >= r.get('update', 0)), None)
        if idx is not None:
            ax[0].axvline(idx, color='#d1495b', ls=':', lw=1.2)
    ax[0].set(title='RLCD training loss' + (f'  ({len(resumes)} resume(s), red)' if resumes else ''),
              xlabel='logged step', ylabel='loss')
    ax[0].legend(fontsize=8)

    if epochs:
        e = sorted(epochs)
        ax[1].plot(e, [epochs[i]['avg_loss'] for i in e], marker='o', color='#1f4e8c')
        ax[1].set_xticks(e)
    ax[1].set(title='average loss per epoch', xlabel='epoch', ylabel='loss')

    ax[2].plot(x, [r['lr_encoder'] for r in steps], color='#1f4e8c', label='encoder LR')
    ax[2].plot(x, [r['lr_head'] for r in steps], color='#e07a1f', label='head LR')
    ax[2].set(title='learning rate (cosine) and exploration noise', xlabel='logged step', ylabel='learning rate')
    ax[2].set_yscale('log')
    if 'sigma' in steps[0]:
        tw = ax[2].twinx()
        tw.plot(x, [r['sigma'] for r in steps], color='#6a994e', ls='--', label='sigma')
        tw.set_ylabel('RLCD sigma')
        tw.legend(loc='lower left', fontsize=8)
    ax[2].legend(loc='upper right', fontsize=8)

    fig.suptitle(f'{Path(model_dir).name}: {len(steps)} logged steps, {len(epochs)} epoch(s) done')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == '__main__':
    print(plot(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'models' / 'LayaLink'))
