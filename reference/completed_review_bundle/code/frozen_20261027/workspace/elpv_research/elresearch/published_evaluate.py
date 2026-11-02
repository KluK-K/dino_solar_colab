"""Predeclared published-method inference views on held-source development only."""
import gc
import numpy as np
import torch

from .common import read_json,save_json,sha256,now,choose_device
from .publication_dense_data import prepare_dense_rows
from .publication_dense_train import native_evaluation
from .publication_train import probability_metrics
from .metrics import classification
from .published_archive import load_published_inference


def evaluate_published(settings,folder,target):
    allowed=set(settings['permitted_target_development_ids'])
    if not target or any(r['partition']!='target_development' or r['id'] not in allowed for r in target):
        raise PermissionError('Unregistered/sealed published-method targets')
    run=read_json(folder/'run.json')
    if {r['id'] for r in target}&set(run['training_ids']+run['validation_ids']+run['unlabelled_source_ids']):
        raise PermissionError('Target image participated in supervised or unlabelled source fitting')
    if sha256(folder/'best.pt')!=run['checkpoint_sha256']:raise RuntimeError('Published-method checkpoint changed')
    destination=folder/'target_development.json'
    if destination.exists():return read_json(destination)
    save_json(folder/'target_development_opening.json',{'at':now(),'run_sha256':sha256(folder/'run.json'),
        'target_ids':[r['id'] for r in target],'scope':'Registered held-source development; no target fitting'})
    examples=prepare_dense_rows(target,settings['input_size']);device=choose_device(settings['device']);results={}
    for view in run['inference_views']:
        model=load_published_inference(run,folder/'best.pt',device,view)
        logits,local=native_evaluation(model,examples,device,settings['batch_size'])
        y=np.array([r['target'] for r in target]);source=run['inference_views'][view]
        results[view]={'metrics':classification(y,logits,source['source_logit_threshold']),'localization':local,
            'source_temperature_calibration':probability_metrics(y,logits,source['source_temperature'])}
        name='target_development_predictions.npz' if view=='primary' else 'target_development_predictions_ensemble.npz'
        np.savez_compressed(folder/name,ids=np.array([r['id'] for r in target]),groups=np.array([r['split_group'] for r in target]),y=y,logits=logits)
        del model;gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
    result={'at':now(),'checkpoint_sha256':run['checkpoint_sha256'],'source_logit_threshold':run['source_logit_threshold'],
        'scope':'Held-source development; primary and ensemble fixed before target scoring; not independent model fits',
        **results['primary'],'inference_views':results}
    save_json(destination,result);return result
