"""Immutable development inference, with explicit disjointness from sealed confirmation."""
from pathlib import Path

import numpy as np
import torch

from .common import ROOT,digest,now,read_json,save_json,sha256,choose_device
from .image_classification import restore_trainable
from .metrics import classification
from .publication_models import build_backbone,PublicationClassifier
from .publication_train import prepare_images,predict,probability_metrics


def evaluate_development(settings,run_folder,rows):
    run_folder=Path(run_folder);run_path=run_folder/'run.json';run=read_json(run_path)
    if not rows:raise ValueError('No development target specified')
    permitted=set(settings['permitted_target_development_ids'])
    if any(r['partition']!='target_development' or r['id'] not in permitted for r in rows):
        raise PermissionError('Only predeclared non-test development targets may be evaluated here')
    for key,source in [('id',run['training_ids']+run['validation_ids'])]:
        if set(source)&{r[key] for r in rows}:raise ValueError('Target identity used in source training')
    if run['checkpoint_sha256']!=sha256(run_folder/'best.pt'):raise RuntimeError('Checkpoint changed')
    identity={'study_id':settings['study_id'],'run_sha256':sha256(run_path),
              'checkpoint_sha256':run['checkpoint_sha256'],'target_ids':[r['id'] for r in rows],
              'image_hashes':[r['image_sha256'] for r in rows],
              'cohort_manifest_sha256':settings['split_manifest']['manifest_sha256'],
              'threshold':run['source_calibrated_logit_threshold'],'temperature':run['source_temperature'],
              'evaluation_code_sha256':sha256(__file__)}
    destination=run_folder/'target_development.json'
    if destination.exists():
        old=read_json(destination)
        if digest(old['identity'])!=digest(identity):raise RuntimeError('Development evaluation identity changed')
        if old['prediction_sha256']!=sha256(run_folder/'target_development_predictions.npz'):
            raise RuntimeError('Saved development predictions changed')
        return old
    save_json(run_folder/'target_development_opening.json',{'opened_at':now(),'identity':identity,
        'role':'development evaluation; not blind confirmation; no target fitting or threshold selection'})
    device=choose_device(settings['device'])
    encoder,dimensions,encoder_identity=build_backbone(run['model']['encoder'],settings['input_size'],device)
    if digest(encoder_identity)!=digest(run['encoder']):raise RuntimeError('Pretrained encoder identity changed')
    model=PublicationClassifier(encoder,dimensions,run['model']['encoder'],run['model']['adaptation'],run['seed']).to(device)
    restore_trainable(model,torch.load(run_folder/'best.pt',map_location='cpu',weights_only=True))
    x=prepare_images(rows,settings['input_size']);logits=predict(model,x,False,device,settings['batch_size'])
    y=np.array([r['target'] for r in rows]);groups=np.array([r['split_group'] for r in rows])
    np.savez_compressed(run_folder/'target_development_predictions.npz',ids=np.array([r['id'] for r in rows]),
                        y=y,logits=logits,groups=groups)
    result={'identity':identity,'completed_at':now(),'metrics':classification(y,logits,identity['threshold']),
            'metrics_at_zero_logit':classification(y,logits,0.),
            'calibration_raw':probability_metrics(y,logits),
            'calibration_source_temperature':probability_metrics(y,logits,identity['temperature']),
            'capture_groups':len(set(groups)),'prediction_sha256':sha256(run_folder/'target_development_predictions.npz'),
            'scope':'Source-only model/threshold/temperature; leave-one-acquisition-source-out DEVELOPMENT.'}
    save_json(destination,result)
    del model,encoder,x
    if device.type=='mps':torch.mps.empty_cache()
    return result
