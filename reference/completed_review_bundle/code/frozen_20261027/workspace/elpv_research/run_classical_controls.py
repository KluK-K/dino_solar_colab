"""Source-calibrated classical controls; CPU-only, no sealed evaluation or new encoder fitting."""
import argparse
import csv
import fcntl
import importlib.metadata
import os
import shutil
import time
import traceback

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from elresearch.common import ROOT,digest,now,read_json,save_json,sha256
from elresearch.data import records
from elresearch.classical_features import HOG_SETTINGS,hog_bank,frozen_dino_bank
from elresearch.classical_training import fit_probe,save_and_verify_probe,evaluate_target_development
from elresearch.publication_splits import fold_records


BASE=ROOT/'artifacts/publication_20261017'
FILES=['classical_features.py','classical_training.py','publication_splits.py','publication_train.py',
       'publication_models.py','image_classification.py','data.py','encoders.py','metrics.py','common.py']


def specification_grid():
    trials=[]
    for representation in ['hog_stats','dinov3_cls']:
        for classifier in ['logistic','linear_svm','rbf_svm']:
            for c in [.001,.01,.1,1.,10.,100.,1000.]:
                for gamma in ([.1,1.,10.] if classifier=='rbf_svm' else [None]):
                    family=representation+'_'+classifier
                    trials.append({'family':family,'representation':representation,'classifier':classifier,
                        'C':c,'gamma_factor':gamma,'candidate_index':len(trials)})
    return trials


def cache_binding(protocol_path,run_path,rows):
    protocol=read_json(protocol_path);run=read_json(run_path)
    bank=frozen_dino_bank(protocol,run,rows)
    return {'protocol_path':str(protocol_path.relative_to(ROOT)),'protocol_sha256':sha256(protocol_path),
        'source_frozen_run_path':str(run_path.relative_to(ROOT)),'source_frozen_run_sha256':sha256(run_path),
        'feature_identity':bank.identity,'feature_bank_sha256':bank.feature_sha256}


def proposal():
    neural=read_json(BASE/'scdd_development_v1/protocol.json')
    ucf=records('ucf','train')+records('ucf','validation')
    scdd=fold_records(neural['split_manifest'],'internal');scdd=scdd[0]+scdd[1]
    ucf_cache=cache_binding(BASE/'calibration_v1/protocol.json',
        BASE/'calibration_v1/runs/dinov3_frozen_lr1_seed11/run.json',ucf)
    scdd_cache=cache_binding(BASE/'scdd_development_v1/protocol.json',
        BASE/'scdd_development_v1/runs/internal/budget_0.05/dinov3_frozen_seed11/run.json',scdd)
    return {'version':1,'kind':'converged classical frozen-feature controls','grid':specification_grid(),
        'calibration_dataset':'ucf','calibration_manifest_sha256':sha256(ROOT/'data/ucf/manifest.json'),
        'source_training_ids':[r['id'] for r in records('ucf','train')],
        'source_validation_ids':[r['id'] for r in records('ucf','validation')],
        'split_manifest':neural['split_manifest'],'neural_design_sha256':sha256(BASE/'scdd_development_v1/design.json'),
        'feature_caches':{'ucf':ucf_cache,'scdd':scdd_cache},
        'hog':HOG_SETTINGS,'augmentation':'all eight fixed D4 views per selected training image; identity view for validation/target',
        'feature_norm':'DINO per-image non-affine LayerNorm eps=1e-5; HOG StandardScaler fit only on selected source training views',
        'regularization':'L2 logistic and squared-hinge linear SVM; RBF gamma = gamma_factor / dimensions; class_weight=None',
        'solver':'liblinear logistic, primal LinearSVC, libsvm RBF; fixed random_state=0; tol=1e-6; CPU thread pools limited to 2',
        'convergence_policy':'LR/linear-SVM max_iter=50000, retry at 200000 on ConvergenceWarning, fail if still unconverged; RBF max_iter=-1',
        'hyperparameter_selection':'highest reused UCF source-validation AUROC per representation/classifier; earliest grid candidate on exact ties',
        'tuning_budget_limit':'Classical grids have 7 C values and 3 RBF gamma factors; NN source calibration had 2 learning-rate candidates. This is a stronger-control extension, not equal tuning budgets or optimization objectives.',
        'source_calibration_trials':70,'development_logical_runs':180,'seeds':[11,22,33],
        'seed_interpretation':'Training-image subsets vary with seed. At 100% the input set and deterministic optimizer are identical; duplicated seed records are not independent optimizer replications.',
        'source_recall':.95,'temperature':'bounded [0.05,20] source-validation margin temperature; no target fit',
        'scope':'Image-only crack-label controls; no pixel supervision, kNN, target fitted statistics, source test or external confirmation.',
        'registration_context':'Continuation of originally planned SVM baselines, added after early internal SCDD development runs; all final confirmation scores remain unopened.',
        'scientific_hashes':{name:sha256(ROOT/'elresearch'/name) for name in FILES},
        'runner_sha256':sha256(__file__),
        'versions':{name:importlib.metadata.version(name) for name in ['numpy','scikit-learn','scikit-image','scipy','torch','joblib']}}


def register(folder):
    settings=proposal();settings['study_id']=digest(settings)[:20]
    path=folder/'protocol.json'
    if path.exists():
        previous=read_json(path)
        if digest({k:v for k,v in previous.items() if k!='registered_at'})!=digest(settings):
            raise RuntimeError('Registered classical-control protocol changed')
        return previous
    settings['registered_at']=now();snapshot=folder/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for name in FILES:shutil.copy2(ROOT/'elresearch'/name,snapshot/name)
    shutil.copy2(__file__,snapshot/'run_classical_controls.py');save_json(path,settings);return settings


def load_banks(settings,which,rows):
    binding=settings['feature_caches'][which]
    protocol=read_json(ROOT/binding['protocol_path']);run=read_json(ROOT/binding['source_frozen_run_path'])
    if sha256(ROOT/binding['protocol_path'])!=binding['protocol_sha256'] or sha256(ROOT/binding['source_frozen_run_path'])!=binding['source_frozen_run_sha256']:
        raise RuntimeError('Source frozen-cache binding changed')
    dino=frozen_dino_bank(protocol,run,rows)
    if dino.feature_sha256!=binding['feature_bank_sha256']:raise RuntimeError('Frozen features changed')
    return {'dinov3_cls':dino,'hog_stats':hog_bank(rows)}


def source_calibration(folder,settings):
    training,validation=records('ucf','train'),records('ucf','validation')
    banks=load_banks(settings,'ucf',training+validation);results=[]
    for spec in settings['grid']:
        path=folder/'calibration_trials'/f"trial_{spec['candidate_index']:02d}";path.mkdir(parents=True,exist_ok=True)
        save_json(folder/'status.json',{'state':'calibrating','at':now(),'pid':os.getpid(),
            'source_calibration_complete':len(results),'source_calibration_planned':70,'current':spec})
        if (path/'run.json').exists():
            result=read_json(path/'run.json')
        else:
            model,result,scores=fit_probe(spec,banks[spec['representation']],training,validation)
            result['study_id']=settings['study_id'];result['scope']='source-only reused UCF calibration; no target conclusion'
            np.savez_compressed(path/'validation_predictions.npz',ids=np.array([r['id'] for r in validation]),
                y=np.array([r['target'] for r in validation]),logits=scores)
            result['prediction_sha256']=sha256(path/'validation_predictions.npz');save_json(path/'run.json',result)
            del model
        if result['study_id']!=settings['study_id']:raise RuntimeError('Calibration result has a different study identity')
        results.append(result)
        print('CALIBRATED',spec['family'],'C',spec['C'],'gamma',spec['gamma_factor'],'source AUROC',result['validation_metrics']['auc'],flush=True)
    winners=[]
    for family in dict.fromkeys(s['family'] for s in settings['grid']):
        winner=max([r for r in results if r['specification']['family']==family],
                   key=lambda r:(r['validation_metrics']['auc'],-r['specification']['candidate_index']))
        spec=winner['specification'];path=folder/'selected_source_models'/family
        if (path/'run.json').exists():stored=read_json(path/'run.json')
        else:
            model,result,scores=fit_probe(spec,banks[spec['representation']],training,validation)
            with np.load(folder/'calibration_trials'/f"trial_{spec['candidate_index']:02d}"/'validation_predictions.npz') as original:
                np.testing.assert_array_equal(scores,original['logits'])
            result['study_id']=settings['study_id'];result['scope']='selected source-only classical calibration model'
            stored=save_and_verify_probe(model,result,scores,banks[spec['representation']],validation,path)
        winners.append({'specification':spec,'source_validation_auroc':stored['validation_metrics']['auc'],
            'source_model_path':str(path.relative_to(ROOT)),'source_run_sha256':sha256(path/'run.json')})
    selection={'study_id':settings['study_id'],'protocol_sha256':sha256(folder/'protocol.json'),'selected':winners,
        'rule':settings['hyperparameter_selection'],'classical_tuning_budget_limit':settings['tuning_budget_limit']}
    destination=folder/'selected_hyperparameters.json'
    if destination.exists():
        if digest(read_json(destination))!=digest(selection):raise RuntimeError('Classical source hyperparameters changed')
    else:save_json(destination,selection)
    return selection


def summarize(folder):
    fields=['fold','representation','classifier','C','gamma_factor','seed','training_fraction','training_images',
        'validation_images','source_validation_auroc','target_development_auroc','target_development_ap','target_recall','target_review_rate']
    rows=[]
    for path in sorted((folder/'runs').glob('*/*/*/run.json')):
        run=read_json(path);spec=run['specification'];target_path=path.parent/'target_development.json'
        target=read_json(target_path)['metrics'] if target_path.exists() else {}
        rows.append({'fold':path.parent.parent.parent.name,'representation':spec['representation'],
            'classifier':spec['classifier'],'C':spec['C'],'gamma_factor':spec['gamma_factor'],'seed':run['seed'],
            'training_fraction':run['training_budget_fraction'],'training_images':run['training_images'],
            'validation_images':run['validation_images'],'source_validation_auroc':run['validation_metrics']['auc'],
            'target_development_auroc':target.get('auc'),'target_development_ap':target.get('average_precision'),
            'target_recall':target.get('recall'),'target_review_rate':target.get('review_rate')})
    with (folder/'development_results.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',default='publication_20261017/classical_controls_v1')
    parser.add_argument('--register-only',action='store_true');parser.add_argument('--calibration-only',action='store_true')
    args=parser.parse_args();folder=ROOT/'artifacts'/args.study;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=register(folder)
    print('REGISTERED',settings['study_id'],'70 source calibration trials; 180 logical development records; CPU only',flush=True)
    if args.register_only:return
    completed=0
    try:
        selection=source_calibration(folder,settings)
        if args.calibration_only:
            save_json(folder/'status.json',{'state':'calibration_complete','at':now(),'source_calibration_complete':70,
                'next':'Resume the same runner without --calibration-only for the registered SCDD controls.'});return
        training,validation,_=fold_records(settings['split_manifest'],'internal')
        banks=load_banks(settings,'scdd',training+validation)
        for fold in settings['split_manifest']['folds']:
            training,validation,target=fold_records(settings['split_manifest'],fold['name'])
            for budget in fold['budgets']:
                for seed in settings['seeds']:
                    for chosen in selection['selected']:
                        spec=chosen['specification'];path=folder/'runs'/fold['name']/f'budget_{budget:g}'/f"{spec['family']}_seed{seed}"
                        save_json(folder/'status.json',{'state':'running','at':now(),'pid':os.getpid(),
                            'source_calibration_complete':70,'completed_runs':completed,'planned_runs':180,
                            'current':{'fold':fold['name'],'budget':budget,'family':spec['family'],'seed':seed}})
                        bank=banks[spec['representation']]
                        if (path/'run.json').exists():
                            result=read_json(path/'run.json')
                            if result['study_id']!=settings['study_id'] or result['checkpoint_sha256']!=sha256(path/'model.joblib'):
                                raise RuntimeError('Completed classical model or study identity changed')
                            model=joblib.load(path/'model.joblib')
                        else:
                            model,result,scores=fit_probe(spec,bank,training,validation,seed,budget,settings['source_recall'])
                            result.update(study_id=settings['study_id'],hyperparameter_selection_sha256=sha256(folder/'selected_hyperparameters.json'),
                                          scope='Real SCDD source/validation only; final confirmation unopened')
                            result=save_and_verify_probe(model,result,scores,bank,validation,path)
                        if target and not (path/'target_development.json').exists():
                            save_json(path/'target_development_opening.json',{'at':now(),'run_sha256':sha256(path/'run.json'),
                                'target_ids':fold['target_development_ids'],'scope':'registered LOSO DEVELOPMENT; no target fitting'})
                            evaluate_target_development(model,result,bank,target,set(fold['target_development_ids']),path)
                        completed+=1;summarize(folder)
                        print('COMPLETED CLASSICAL',fold['name'],budget,seed,spec['family'],'source AUROC',result['validation_metrics']['auc'],flush=True)
                        del model
        save_json(folder/'status.json',{'state':'complete','at':now(),'completed_runs':completed,'planned_runs':180,
            'source_calibration_complete':70,'full_budget_repetition_limit':settings['seed_interpretation']})
    except BaseException as error:
        save_json(folder/'status.json',{'state':'failed','at':now(),'pid':os.getpid(),'completed_runs':completed,
            'error':str(error),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
