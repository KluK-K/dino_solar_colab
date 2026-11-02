"""Separate frozen confirmation gate for complete dense architectures."""
import time
import numpy as np
import torch
from transformers import DINOv3ViTConfig,DINOv3ViTModel

from .common import ROOT,read_json,save_json,sha256,choose_device,now
from .data import records
from .full_models import FullDINO,FullDeepLab
from .heads import NormalBank
from .dense_train import prepare_dense,dense_predict,native_localization
from .metrics import classification

def code_hashes():
    return {name:sha256(ROOT/'elresearch'/name) for name in
            ['data.py','encoders.py','full_models.py','dense_train.py','dense_evaluate.py','heads.py','adapt.py','metrics.py','classification_heads.py']}

def freeze_dense(run_id):
    folder=ROOT/'artifacts/dense_runs'/run_id;run=read_json(folder/'run.json')
    frozen={'created_at':now(),'run_sha256':sha256(folder/'run.json'),'checkpoint_sha256':sha256(folder/'model.pt'),
            'normal_bank_sha256':sha256(folder/'normal_bank.pt') if run['reference'] else None,
            'scientific_hashes':code_hashes(),'source_manifest':run['source_manifest']}
    destination=folder/'frozen.json'
    if destination.exists():
        old=read_json(destination)
        if any(old.get(k)!=v for k,v in frozen.items() if k!='created_at'):
            raise RuntimeError('Frozen full-model protocol cannot be replaced')
        return old
    if frozen['checkpoint_sha256']!=run['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed after training')
    save_json(destination,frozen);return frozen

def evaluate_dense(run_id,dataset=None,allow_provisional=False,device='auto'):
    folder=ROOT/'artifacts/dense_runs'/run_id;run=read_json(folder/'run.json');frozen=read_json(folder/'frozen.json')
    if (sha256(folder/'run.json')!=frozen['run_sha256'] or sha256(folder/'model.pt')!=frozen['checkpoint_sha256']
            or code_hashes()!=frozen['scientific_hashes']):
        raise RuntimeError('Frozen run, model or scientific code changed')
    if run['reference'] and sha256(folder/'normal_bank.pt')!=frozen['normal_bank_sha256']:
        raise RuntimeError('Frozen normal reference bank changed')
    source=read_json(ROOT/'data'/run['dataset']/'manifest.json')
    if source['signature']!=frozen['source_manifest']:raise RuntimeError('Source cohort changed')
    if not run['publication_eligible'] and not allow_provisional:
        raise PermissionError('Unverified source provenance requires explicit pilot evaluation')
    dataset=dataset or run['dataset']
    if dataset=='elpv':raise PermissionError('Course test remains sealed')
    partition='test' if dataset==run['dataset'] else 'external'
    rows=records(dataset,partition,evaluation=True)
    target=read_json(ROOT/'data'/dataset/'manifest.json')
    if dataset!=run['dataset']:
        hashes={r['pixel_sha256'] for r in source['records']}
        if any(r['pixel_sha256'] in hashes for r in rows):raise ValueError('Cross-domain image overlap')
    ledger=folder/f'evaluation_{dataset}.json'
    if ledger.exists():raise RuntimeError('Domain already opened; retain its ledger and results')
    device=choose_device(device)
    if run['architecture']=='dinov3_decoder':
        backbone=DINOv3ViTModel(DINOv3ViTConfig.from_dict(run['encoder']['config']))
        bank=NormalBank.load(folder/'normal_bank.pt') if run['reference'] else None
        if bank is not None and bank.encoder_key!=run['encoder_key']:raise RuntimeError('Bank encoder mismatch')
        c=run['reference_config']
        model=FullDINO(backbone,bank,run['adaptation'],reference_neighbors=c['neighbors'],geometry_weight=c['geometry_weight'],top_fraction=run['top_fraction'],image_head=run.get('image_head',{}).get('kind','linear'))
    else:
        model=FullDeepLab(pretrained=False,top_fraction=run['top_fraction'])
    # Random constructor tensors are completely overwritten before any image is evaluated.
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False).to(device)
    save_json(ledger,{'opened_at':now(),'target_manifest':target['signature'],'state':'opened',
                      'frozen_run':sha256(folder/'frozen.json'),'selection_on_target':False})
    started=time.perf_counter();examples=prepare_dense(rows,run['input_size'])
    scores,maps=dense_predict(model,examples,device,run['batch_size'])
    result={'run_id':run_id,'dataset':dataset,'partition':partition,'completed_at':now(),
            'metrics':classification([r['target'] for r in rows],scores,run['threshold']),
            'localization':native_localization(examples,maps,pixel_threshold=.5),
            'threshold_origin':'source validation; no target recalibration','seconds_preprocessing_and_inference':time.perf_counter()-started,
            'publication_eligible':bool(run['publication_eligible'] and target['publication_eligible']),
            'group_semantics':target.get('group_semantics'),'scope':'confirmed protocol' if run['publication_eligible'] else 'provenance-unverified pilot'}
    np.savez_compressed(folder/f'predictions_{dataset}.npz',ids=np.array([r['id'] for r in rows]),
                        y=np.array([r['target'] for r in rows]),scores=scores,groups=np.array([r['split_group'] for r in rows]))
    save_json(folder/f'results_{dataset}.json',result)
    save_json(ledger,{**read_json(ledger),'state':'completed','results_sha256':sha256(folder/f'results_{dataset}.json')})
    return result
