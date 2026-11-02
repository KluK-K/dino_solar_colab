"""Frozen mechanism-control systems on predeclared held-source development."""
import gc
import numpy as np
import torch

from .common import read_json,save_json,sha256,now,choose_device,digest
from .image_classification import restore_trainable
from .metrics import classification
from .publication_train import probability_metrics
from .publication_dense_data import prepare_dense_rows
from .publication_dense_train import native_evaluation
from .packed_normal_bank import load_reference_bank
from .mechanistic_models import build_mechanistic_system
from .mechanistic_train import scientific_hashes


def restore_mechanistic(model,spec,path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if spec['architecture']=='dinov3_normal_distance':model.load_calibration_state(state)
    else:restore_trainable(model,state)


def evaluate_mechanistic(settings,folder,target):
    allowed=set(settings['permitted_target_development_ids'])
    if not target or any(r['partition']!='target_development' or r['id'] not in allowed for r in target):
        raise PermissionError('Unregistered or sealed mechanism-control target')
    if scientific_hashes()!=settings['scientific_hashes']:raise RuntimeError('Mechanism-control implementation changed')
    run=read_json(folder/'run.json')
    if {r['id'] for r in target}&set(run['training_ids']+run['validation_ids']):raise PermissionError('Target participated in source fitting')
    if sha256(folder/'best.pt')!=run['checkpoint_sha256']:raise RuntimeError('Mechanism-control checkpoint changed')
    destination=folder/'target_development.json'
    if destination.exists():return read_json(destination)
    save_json(folder/'target_development_opening.json',{'at':now(),'run_sha256':sha256(folder/'run.json'),
        'target_ids':[r['id'] for r in target],'scope':'Registered source-held-out development; final confirmation unopened'})
    device=choose_device(settings['device'])
    bank=load_reference_bank(folder/'normal_bank_recipe.json',run['training_ids']) if run['model'].get('reference') else None
    model,identity,_=build_mechanistic_system(run['model'],run['seed'],settings['input_size'],device,bank=bank)
    if digest(identity)!=digest(run['encoder']):raise RuntimeError('Frozen mechanism-control encoder changed')
    restore_mechanistic(model,run['model'],folder/'best.pt')
    examples=prepare_dense_rows(target,settings['input_size'])
    logits,local=native_evaluation(model,examples,device,settings['batch_size'])
    y=np.array([r['target'] for r in target])
    result={'at':now(),'checkpoint_sha256':run['checkpoint_sha256'],
        'scope':'Predeclared held-source development; no target fitting',
        'source_logit_threshold':run['source_logit_threshold'],
        'metrics':classification(y,logits,run['source_logit_threshold']),'localization':local,
        'source_temperature_calibration':probability_metrics(y,logits,run['source_temperature'])}
    np.savez_compressed(folder/'target_development_predictions.npz',ids=np.array([r['id'] for r in target]),
        groups=np.array([r['split_group'] for r in target]),y=y,logits=logits)
    save_json(destination,result);del model,bank,examples;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    return result
