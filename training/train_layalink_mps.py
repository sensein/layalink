"""Laya RLCD ontology fine-tuning on Apple Silicon MPS or CPU.
Run from the repository root: python training/train_layalink_mps.py --device auto
Uses the same canonical training objective as upstream notebooks, with separate splits.
"""
import argparse, json, os, random
from pathlib import Path
os.environ.setdefault('USE_TF','0')
# Must be set before importing torch. Unsupported MPS operations can fall back to CPU.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--datasets',default='all',help="comma list of data/train/<name> folders, or 'all'")
    p.add_argument('--model',default='convaiinnovations/laya')
    p.add_argument('--revision',default=None)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent.parent/'models'/'LayaLink')
    p.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--micro-batch',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=32)
    p.add_argument('--max-len',type=int,default=1024)
    p.add_argument('--head-max-len',type=int,default=256)
    p.add_argument('--loss',choices=['rlcd','soft-ce'],default='rlcd')
    p.add_argument('--freeze-encoder',action='store_true')
    p.add_argument('--no-checkpointing',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='Validate JSONL/splits only; no torch or model download')
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    from load_data import load_splits,preprocess
    rows=load_splits(args.data_root,args.datasets)
    print({s:{'cases':len(rs),'decisions':sum(len(r['questions']) for r in rs)} for s,rs in rows.items()})
    if args.dry_run:return
    import torch
    import numpy as np
    from huggingface_hub import snapshot_download,model_info
    from laya.train import TrainConfig,load_checkpoint,train_model,calibration_records,evaluate_records,save_checkpoint
    from laya.calibrate import fit_temperature_map
    available=hasattr(torch.backends,'mps') and torch.backends.mps.is_available()
    if args.device=='mps' and not available:raise RuntimeError('MPS requested but unavailable; choose --device cpu')
    device=torch.device('mps' if args.device=='auto' and available else 'cpu' if args.device=='auto' else args.device)
    print('Device:',device,'; float32 training; CUDA AMP disabled')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if Path(args.model).is_dir():model_dir=args.model;revision='local'
    else:
        revision=model_info(args.model,revision=args.revision).sha
        model_dir=snapshot_download(args.model,revision=revision)
    model,tok,cfg=load_checkpoint(model_dir)
    items=preprocess(tok,rows,args.max_len,args.head_max_len)
    tc=TrainConfig(epochs=args.epochs,micro_batch=args.micro_batch,grad_accum=args.grad_accum,
        encoder_lr=2.5e-5,head_lr=1e-4,loss=args.loss,rl_samples=4,
        sigma_start=.4,sigma_end=.1,shuffle_options=('choice',),seed=42,
        amp=False,gradient_checkpointing=not args.no_checkpointing,
        freeze_encoder=args.freeze_encoder,max_len=args.max_len,head_max_len=args.head_max_len,log_every=20)
    tc.validate();args.out.mkdir(parents=True,exist_ok=True)
    out_cfg=dict(cfg,max_len=args.max_len,head_max_len=args.head_max_len)
    logdir=args.out/'training';logdir.mkdir(parents=True,exist_ok=True)
    def log(**rec):
        import time as _t
        with open(logdir/'training_log.jsonl','a') as f:f.write(json.dumps(dict(rec,time=round(_t.time(),1)),default=float)+'\n')
    log(event='start',epoch=0,update=0,device=str(device),epochs=args.epochs,datasets=args.datasets,
        train_items=len(items['train']),effective_batch=args.micro_batch*args.grad_accum)
    def checkpoint(epoch,loss):
        log(event='epoch_end',epoch=epoch+1,avg_loss=float(loss))
        save_checkpoint(model,tok,out_cfg,str(args.out/'checkpoint_latest'))
        (args.out/'checkpoint_latest'/'progress.json').write_text(json.dumps({'completed_epochs':epoch+1,'loss':loss}))
    history=train_model(model,tok,items['train'],tc,device,args.max_len,args.head_max_len,on_epoch_end=checkpoint)
    cal=calibration_records(model,tok,items['calibration'],device,args.max_len,args.head_max_len,batch_size=1)
    fitted=fit_temperature_map(cal)
    out_cfg.update(temperature=fitted['temperature'],fine_tuned=True,model_name='LayaLink')
    out_cfg.pop('temperature_by_options',None)
    if fitted.get('temperature_by_options'):out_cfg['temperature_by_options']=fitted['temperature_by_options']
    save_checkpoint(model,tok,out_cfg,str(args.out))
    test=calibration_records(model,tok,items['test'],device,args.max_len,args.head_max_len,batch_size=1)
    report={'device':str(device),'model_revision':revision,'datasets':args.datasets,
       'official_oaei_evaluation':False,'loss':args.loss,'history':history,
       'raw':evaluate_records(test),'calibrated':evaluate_records(test,out_cfg['temperature'],out_cfg.get('temperature_by_options'))}
    (args.out/'evaluation.json').write_text(json.dumps(report,indent=2,default=float))
    (args.out/'run_config.json').write_text(json.dumps(vars(args),indent=2,default=str))
    log(event='done',temperature=out_cfg['temperature'])
    (logdir/'training_summary.json').write_text(json.dumps({'datasets':args.datasets,'epochs':args.epochs,'history':history,
        'temperature':out_cfg['temperature'],'test':{'raw':report['raw'],'calibrated':report['calibrated']}},indent=2,default=float))
    try:                                           # loss curves; never fails the run
        import sys as _s;_s.path.insert(0,str(Path(__file__).resolve().parent))
        from plot_training import plot
        print('training curves:',plot(args.out))
    except Exception as e:print(f'(plotting skipped: {type(e).__name__}: {e})')
    print(json.dumps(report,indent=2,default=float))
    print('Saved:',args.out,'; load with laya.Agent(str(path), device="mps")')
if __name__=='__main__':main()
