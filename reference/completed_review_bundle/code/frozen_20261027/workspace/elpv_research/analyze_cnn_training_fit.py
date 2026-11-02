"""CPU-only evaluation-mode fit diagnostics for the nine finished CNN source runs."""
import gc
import os
import time

import numpy as np
import torch

from elresearch.common import ROOT,read_json,save_json,digest,sha256,now
from elresearch.image_classification import restore_trainable
from elresearch.publication_models import build_backbone,PublicationClassifier
from elresearch.publication_train import prepare_images,predict,probability_metrics
from elresearch.source_diagnostics import source_summary


def main():
    torch.set_num_threads(2);base=ROOT/'artifacts/publication_20261017'
    folder=base/'cnn_training_fit_diagnostic';folder.mkdir(exist_ok=True)
    rows={r['id']:r for r in read_json(ROOT/'data/scdd_publication_v1/manifest.json')['records'] if r['partition'] in {'train','validation'}}
    jobs=sorted((base/'scdd_development_v1/runs/internal/budget_1').glob('*net*_full_seed*/run.json'))
    jobs=[p for p in jobs if not read_json(p)['model']['encoder'].startswith('dinov3')]
    if len(jobs)!=9:raise RuntimeError('All three CNNs and three full-label source seeds must be complete before this diagnostic')
    results=[]
    for path in jobs:
        run=read_json(path);destination=folder/(path.parent.name+'.json')
        save_json(folder/'status.json',{'state':'running','at':now(),'pid':os.getpid(),'completed_runs':len(results),'planned_runs':9,'current':path.parent.name})
        if destination.exists():
            result=read_json(destination)
            if result['source_run_sha256']!=sha256(path):raise RuntimeError('Diagnostic source run changed')
            results.append(result);continue
        if sha256(path.parent/'best.pt')!=run['checkpoint_sha256']:raise RuntimeError('Source CNN checkpoint changed')
        started=time.perf_counter();spec=run['model'];device=torch.device('cpu')
        encoder,dim,identity=build_backbone(spec['encoder'],336,device)
        if digest(identity)!=digest(run['encoder']):raise RuntimeError('Source CNN encoder identity changed')
        model=PublicationClassifier(encoder,dim,spec['encoder'],spec['adaptation'],run['seed']).eval()
        restore_trainable(model,torch.load(path.parent/'best.pt',map_location='cpu',weights_only=True))
        groups={}
        for key,ids in [('training',run['training_ids']),('validation',run['validation_ids'])]:
            examples=[rows[i] for i in ids]
            if any(r['partition'] not in {'train','validation'} for r in examples):raise PermissionError('No confirmation access')
            values=prepare_images(examples,336)
            logits=predict(model,values,False,device,8);y=np.array([r['target'] for r in examples])
            groups[key]={'ranking':source_summary(y,logits,[r['acquisition_source'] for r in examples]),
                         'raw_probability_metrics':probability_metrics(y,logits),'logit_sd':float(np.std(logits))}
            del values
        history=read_json(path.parent/'history.json');best=history[run['best_round']-1]
        result={'at':now(),'model':spec['name'],'seed':run['seed'],'source_run_sha256':sha256(path),
                'source_run':str(path.relative_to(ROOT)),'checkpoint_sha256':run['checkpoint_sha256'],
                'best_round':run['best_round'],'training_step_BCE_at_best':best['training_bce'],
                'minimum_training_step_BCE':min(h['training_bce'] for h in history),
                'evaluation_mode':groups,'device':'cpu','seconds':time.perf_counter()-started,
                'scope':'Previously observed source images only; evaluation mode has no D4/stochastic-depth noise. CPU numerical differences from MPS may exist; this does not replace registered source reload verification.',
                'interpretation':'Training-step and evaluation losses differ in augmentation/stochastic behavior. Inspect training and validation ranking/calibration jointly; do not infer a causal optimization failure from one loss value.'}
        save_json(destination,result);results.append(result)
        print('CNN FIT DIAGNOSTIC',spec['name'],run['seed'],'train AUROC',groups['training']['ranking']['pooled_auroc'],
              'val AUROC',groups['validation']['ranking']['pooled_auroc'],flush=True)
        del model,encoder;gc.collect()
    save_json(folder/'summary.json',{'at':now(),'runs':results,'analysis_code_sha256':sha256(__file__),'confirmation_scores_opened':False})
    save_json(folder/'status.json',{'state':'complete','at':now(),'pid':os.getpid(),'completed_runs':9,'planned_runs':9})


if __name__=='__main__':main()
