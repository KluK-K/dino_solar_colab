"""Pinned frozen-representation/resolution/scale controls; no sealed predictions."""
import argparse
import csv
import fcntl
import importlib.metadata
import os
import shutil
import time
import traceback
from pathlib import Path

import joblib
import torch

from elresearch.common import ROOT,digest,now,read_json,save_json,sha256
from elresearch.data import records
from elresearch.classical_training import save_and_verify_probe,evaluate_target_development
from elresearch.publication_splits import fold_records
from elresearch.representation_features import official_identities,cached_banks
from elresearch.representation_training import fit_representation_probe


BASE=ROOT/'artifacts/publication_20261017'
PARENT=BASE/'cnn_stability_v1'
FILES=['representation_features.py','representation_training.py','classical_features.py','classical_training.py',
       'publication_splits.py','publication_train.py','publication_models.py','image_classification.py',
       'encoders.py','data.py','metrics.py','common.py','adapt.py','classification_heads.py']


def feature_specifications():
    values=[]
    for encoder,sizes in [('dinov3_small',[224,336,448]),('dinov3_base',[336]),('resnet18',[336])]:
        for size in sizes:
            for pooling in (['global'] if encoder=='resnet18' else ['cls','patch_mean','concat']):
                values.append({'feature_key':f'{encoder}_{size}_{pooling}',
                    'encoder':encoder,'input_size':size,'pooling':pooling})
    return values


def specification_grid():
    trials=[]
    for feature in feature_specifications():
        for classifier in (['logistic','linear_svm','rbf_svm'] if feature['encoder']=='resnet18' else ['logistic']):
            cs=[.01,.1,1.] if classifier=='logistic' else ([.01,.1,1.,10.] if classifier=='linear_svm' else [1.,10.])
            for c in cs:
                for gamma in ([.1,1.,10.] if classifier=='rbf_svm' else [None]):
                    trials.append({**feature,'family':feature['feature_key']+'_'+classifier,
                        'representation':feature['feature_key'],'classifier':classifier,
                        'C':c,'gamma_factor':gamma,'candidate_index':len(trials)})
    return trials


def proposal():
    neural=read_json(BASE/'scdd_development_v1/protocol.json')
    split=neural['split_manifest'];features=feature_specifications();grid=specification_grid()
    families=len({s['family'] for s in grid})
    unique_fits=sum((3 if b<1 else 1) for f in split['folds'] for b in f['budgets'])*families
    return {'version':1,'kind':'Frozen representation, input resolution and model scale controls',
        'registered_after':'72 internal image-model fits and first held-source development results; all final confirmation unopened',
        'feature_specifications':features,'grid':grid,'encoder_identities':official_identities(),
        'source_training_ids':[r['id'] for r in records('ucf','train')],
        'source_validation_ids':[r['id'] for r in records('ucf','validation')],
        'source_manifest_sha256':sha256(ROOT/'data/ucf/manifest.json'),'split_manifest':split,
        'neural_protocol_sha256':sha256(BASE/'scdd_development_v1/protocol.json'),
        'calibration_trials':len(grid),'development_unique_runs':unique_fits,
        'planned_runs':len(grid)+unique_fits,'families':families,'seed_policy':{'subset_seeds':[11,22,33],'full_budget_seeds':[11]},
        'feature_augmentation':'All eight deterministic D4 views per training image; identity view at validation and held-source prediction',
        'pooling':'CLS, image-area-weighted mean of non-register patches, or concatenation, followed by per-image non-affine LayerNorm. ResNet18 uses global average pooling.',
        'controlled_comparisons':'Same three-C logistic grid for all13 encoder/size/pooling representations. Additional linear/RBF SVMs on frozen ImageNet ResNet18 connect the original feature-probe plan.',
        'solver':'L2 liblinear logistic; LinearSVC dual=auto; RBF libsvm gamma=factor/d; tol1e-4; max_iter10000 then50000 retry for finite solvers; fail on unresolved ConvergenceWarning; CPU2 threads',
        'source_selection':'Highest UCF validation AUROC per family, earliest candidate on exact ties. No SCDD label chooses C/gamma.',
        'selection_exposure':'UCF430 training+144 validation labels are development exposure, including for later SCDD-fitted models. Do not call SCDD-to-UCF a fully unseen-dataset experiment.',
        'source_recall':.95,'feature_batch_size':4,'device':'mps','dtype':'float32',
        'independent_replication_limit':'Probe fitting is deterministic; full-budget results saved once. Three lower-budget seeds vary annotation subsets, not encoder pretraining or stochastic optimization.',
        'resolution_limit':'Bicubic interpolation changes processing scale, not recovered physical detail. Patch averaging uses fractional geometric image area; boundary tokens can contain padding context.',
        'supervision':'Source image labels only; feature extraction never receives masks or labels; no DINOv2, target-fitted scaler, kNN or encoder adaptation.',
        'tuning_budget_limit':'This49-trial sensitivity grid differs from the wider old70-trial classical grid and two-rate neural calibration. It is not an equal optimization-budget claim.',
        'scientific_hashes':{name:sha256(ROOT/'elresearch'/name) for name in FILES},
        'runner_sha256':sha256(__file__),
        'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','numpy','scikit-learn','scipy','joblib']}}


def register(folder):
    settings=proposal();settings['study_id']=digest(settings)[:20];path=folder/'protocol.json'
    if path.exists():
        old=read_json(path)
        if digest({k:v for k,v in old.items() if k!='registered_at'})!=digest(settings):
            raise RuntimeError('Registered representation/scale protocol changed')
        return old
    snapshot=folder/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for name in FILES:shutil.copy2(ROOT/'elresearch'/name,snapshot/name)
    shutil.copy2(__file__,snapshot/Path(__file__).name)
    settings['registered_at']=now();save_json(path,settings);return settings


def check_implementation(settings):
    if {n:sha256(ROOT/'elresearch'/n) for n in FILES}!=settings['scientific_hashes']:
        raise RuntimeError('Registered representation implementation changed')
    if settings['runner_sha256']!=sha256(__file__):raise RuntimeError('Registered representation runner changed')
    if sha256(ROOT/'data/ucf/manifest.json')!=settings['source_manifest_sha256']:
        raise RuntimeError('Registered UCF source manifest changed')


def load_or_fit(settings,spec,bank,training,validation,seed,budget,folder):
    if (folder/'run.json').exists():
        result=read_json(folder/'run.json')
        if (result['study_id']!=settings['study_id'] or result['specification']!=spec
                or result['checkpoint_sha256']!=sha256(folder/'model.joblib')
                or result['feature_bank_sha256']!=bank.feature_sha256):
            raise RuntimeError('Registered fitted representation model changed')
        return joblib.load(folder/'model.joblib'),result
    check_implementation(settings)
    model,result,scores=fit_representation_probe(spec,bank,training,validation,seed,budget,settings['source_recall'])
    result.update(study_id=settings['study_id'],scientific_hashes=settings['scientific_hashes'],
        scope='Image-label-only development; all final confirmation unopened')
    result=save_and_verify_probe(model,result,scores,bank,validation,folder)
    return model,result


def summarize(folder):
    values=[]
    for path in sorted((folder/'runs').glob('*/*/*/run.json')):
        run=read_json(path);target=path.parent/'target_development.json'
        metric=read_json(target)['metrics'] if target.exists() else {}
        values.append({'fold':path.parents[2].name,'budget':run['training_budget_fraction'],
            'family':run['specification']['family'],'seed':run['seed'],'training_images':run['training_images'],
            'source_auroc':run['validation_metrics']['auc'],'target_auroc':metric.get('auc'),
            'target_recall':metric.get('recall'),'target_review_rate':metric.get('review_rate')})
    if values:
        with (folder/'development_results.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--wait-for-cnn-stability',action='store_true');args=parser.parse_args()
    folder=BASE/'representation_scale_v1';folder.mkdir(exist_ok=True)
    lock=(folder/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=register(folder);print('REPRESENTATION CONTROLS REGISTERED',settings['study_id'],settings['planned_runs'],'unique fits',flush=True)
    if args.register_only:return
    completed=0
    def status(state,**extra):
        save_json(folder/'status.json',{'state':state,'at':now(),'pid':os.getpid(),
            'completed_runs':completed,'planned_runs':settings['planned_runs'],**extra})
    try:
        while True:
            prior=read_json(PARENT/'status.json')
            if prior['state']=='complete':
                with (PARENT/'runner.lock').open('a') as parent_lock:fcntl.flock(parent_lock,fcntl.LOCK_EX)
                break
            if not args.wait_for_cnn_stability or prior['state'] in {'failed','bounded_stop'}:
                raise RuntimeError('Prior CNN stability study needs completion/resume')
            status('queued_after_cnn_stability',parent_state=prior['state']);time.sleep(30)
        check_implementation(settings);torch.set_num_threads(2)
        ucf_train,ucf_val=records('ucf','train'),records('ucf','validation')
        train,val,_=fold_records(settings['split_manifest'],'internal');all_rows=ucf_train+ucf_val+train+val
        banks={}
        with (BASE/'gpu_work_queue.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            for encoder,size in dict.fromkeys((s['encoder'],s['input_size']) for s in settings['feature_specifications']):
                if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Less than3GiB free before feature preparation')
                status('preparing_features',current={'encoder':encoder,'size':size})
                pools=[s['pooling'] for s in settings['feature_specifications'] if (s['encoder'],s['input_size'])==(encoder,size)]
                prepared=cached_banks(all_rows,settings['encoder_identities'][encoder],size,pools,
                    settings['device'],settings['feature_batch_size'])
                for pooling,bank in prepared.items():banks[f'{encoder}_{size}_{pooling}']=bank
        # Fitting is CPU-only; release the GPU queue after encoder caches are complete.
        trials=[]
        for spec in settings['grid']:
            status('calibrating',current=spec)
            path=folder/'calibration_trials'/f"trial_{spec['candidate_index']:02d}"
            model,result=load_or_fit(settings,spec,banks[spec['feature_key']],ucf_train,ucf_val,11,1.,path)
            trials.append((result,path));completed+=1;del model
        selected=[]
        for family in dict.fromkeys(s['family'] for s in settings['grid']):
            result,path=max([p for p in trials if p[0]['specification']['family']==family],
                key=lambda p:(p[0]['validation_metrics']['auc'],-p[0]['specification']['candidate_index']))
            selected.append({'specification':result['specification'],'source_run_path':str(path.relative_to(ROOT)),
                'source_run_sha256':sha256(path/'run.json'),'source_validation_auroc':result['validation_metrics']['auc']})
        selection={'study_id':settings['study_id'],'protocol_sha256':sha256(folder/'protocol.json'),'selected':selected}
        selection_path=folder/'selected_hyperparameters.json'
        if selection_path.exists() and read_json(selection_path)!=selection:raise RuntimeError('Source feature selection changed')
        save_json(selection_path,selection)
        for fold in settings['split_manifest']['folds']:
            training,validation,target=fold_records(settings['split_manifest'],fold['name'])
            for budget in fold['budgets']:
                for seed in ([11,22,33] if budget<1 else [11]):
                    for chosen in selected:
                        spec=chosen['specification'];status('running',current={'fold':fold['name'],'budget':budget,'seed':seed,'family':spec['family']})
                        path=folder/'runs'/fold['name']/f'budget_{budget:g}'/f"{spec['family']}_seed{seed}"
                        bank=banks[spec['feature_key']]
                        model,result=load_or_fit(settings,spec,bank,training,validation,seed,budget,path)
                        if target and not (path/'target_development.json').exists():
                            save_json(path/'target_development_opening.json',{'at':now(),'run_sha256':sha256(path/'run.json'),
                                'target_ids':fold['target_development_ids'],'scope':'Registered source-held-out DEVELOPMENT, not sealed confirmation'})
                            evaluate_target_development(model,result,bank,target,set(fold['target_development_ids']),path)
                        completed+=1;summarize(folder);del model
        if completed!=settings['planned_runs']:raise RuntimeError('Completed fit count differs from registered plan')
        status('complete',source_calibration_complete=settings['calibration_trials'],
            next='Review source/development evidence, verify all original-scope controls, then freeze confirmation separately')
    except BaseException as error:
        status('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':
    main()
