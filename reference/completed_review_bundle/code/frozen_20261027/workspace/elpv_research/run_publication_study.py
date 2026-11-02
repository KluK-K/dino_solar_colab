"""Real-domain development after source-only calibration; sealed data are never inferred here."""
import argparse
import csv
import fcntl
import os
import shutil
import time
import traceback

from elresearch.common import ROOT,digest,now,read_json,save_json,sha256
from elresearch.publication_evaluate import evaluate_development
from elresearch.publication_splits import partition_manifest,fold_records
from elresearch.publication_train import FILES,scientific_hashes,train


EXTRA_FILES=['publication_splits.py','publication_evaluate.py']
CALIBRATION=ROOT/'artifacts/publication_20261017/calibration_v1'


def design():
    calibration=read_json(CALIBRATION/'protocol.json')
    return {'version':1,'kind':'label-limited and leave-one-acquisition-source-out development',
        'split_manifest':partition_manifest(),'seeds':[11,22,33],
        'families':list(dict.fromkeys(s['family'] for s in calibration['specifications'])),
        'calibration_protocol_sha256':sha256(CALIBRATION/'protocol.json'),
        'hyperparameter_binding_rule':'only selected_hyperparameters.json produced by the registered source-only calibration',
        'input_size':336,'batch_size':8,'updates_per_round':50,'maximum_rounds':40,'patience':6,
        'weight_decay':1e-4,'target_source_recall':.95,'device':'auto',
        'scientific_hashes':scientific_hashes(),
        'extra_scientific_hashes':{name:sha256(ROOT/'elresearch'/name) for name in EXTRA_FILES},
        'runner_sha256':sha256(__file__),
        'augmentation_and_optimizer':'same D4, AdamW, cosine schedule and BN policy as registered source calibration',
        'annotation_cost':'nested class-stratified training images per seed; full source validation labels counted separately and in total',
        'primary_readouts':['image AUROC','image average precision','macro F1 and balanced accuracy at source 95%-recall threshold',
                            'target recall and review rate','Brier and NLL before/after source temperature scaling'],
        'paired_comparisons':'DINO LoRA versus frozen DINO and each fully fine-tuned CNN; full DINO as adaptation control',
        'statistics_plan':'report all three seed values and sample SD; paired acquisition-family bootstrap on development predictions; seed variability is not a confidence interval; no pooled patch-level pseudo-replicates',
        'claim_policy':'Results are development evidence. No superiority or cross-dataset confirmation claim before a separate frozen confirmation registry.',
        'scope':'Real SCDD images; annotated crack presence anywhere in field of view, not central-cell failure. Laboratory/source labels do not prove physical module independence.',
        'pretraining_limit':'DINOv3 and CNNs have different pretraining data and objectives. Compare complete transfer systems; do not attribute differences solely to Transformer versus CNN architecture.',
        'sealed_policy':{'scdd_test':117,'ucf_supplemental':507,'elpv_course_test':'never opened by this pipeline',
                         'automatic_confirmation_evaluation':False},
        'planned_runs':180}


def register(folder):
    proposal=design();proposal['design_id']=digest(proposal)[:20]
    path=folder/'design.json'
    if path.exists():
        previous=read_json(path)
        if digest({k:v for k,v in previous.items() if k!='registered_at'})!=digest(proposal):
            raise RuntimeError('Registered real-domain development design changed')
        return previous
    proposal['registered_at']=now();snapshot=folder/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for name in FILES+EXTRA_FILES:shutil.copy2(ROOT/'elresearch'/name,snapshot/name)
    shutil.copy2(__file__,snapshot/'run_publication_study.py')
    save_json(path,proposal);return proposal


def bind_hyperparameters(folder,registered):
    selection_path=CALIBRATION/'selected_hyperparameters.json'
    if read_json(CALIBRATION/'status.json')['state']!='complete':raise RuntimeError('Calibration is not complete')
    selected=read_json(selection_path)
    if selected['protocol_sha256']!=registered['calibration_protocol_sha256']:
        raise RuntimeError('Calibration source changed')
    specifications=[]
    for winner in selected['selected']:
        spec=winner['specification'];specifications.append({**spec,'name':spec['family']})
    if sorted(s['family'] for s in specifications)!=sorted(registered['families']):raise ValueError('Family support changed')
    settings={**registered,'specifications':specifications,'hyperparameter_selection_sha256':sha256(selection_path),
              'permitted_target_development_ids':sorted({i for f in registered['split_manifest']['folds']
                                                         for i in f['target_development_ids']})}
    settings['study_id']=digest(settings)[:20]
    path=folder/'protocol.json'
    if path.exists():
        previous=read_json(path)
        if digest({k:v for k,v in previous.items() if k!='bound_at'})!=digest(settings):
            raise RuntimeError('Bound real-domain protocol changed')
        return previous
    settings['bound_at']=now();save_json(path,settings);return settings


def summarize(folder):
    columns=['fold','training_fraction','model','seed','training_images','validation_images','total_labeled_images',
        'source_validation_auroc','target_development_auroc','target_development_ap','target_recall','target_review_rate',
        'best_round','trainable_parameters']
    rows=[]
    for path in sorted((folder/'runs').glob('*/*/*/run.json')):
        run=read_json(path);target_path=path.parent/'target_development.json'
        target=read_json(target_path)['metrics'] if target_path.exists() else {}
        rows.append({'fold':path.parent.parent.parent.name,'training_fraction':run['training_budget_fraction'],
            'model':run['model']['name'],'seed':run['seed'],'training_images':run['training_images'],
            'validation_images':run['validation_images'],'total_labeled_images':run['total_labeled_images_including_validation'],
            'source_validation_auroc':run['validation_metrics']['auc'],'target_development_auroc':target.get('auc'),
            'target_development_ap':target.get('average_precision'),'target_recall':target.get('recall'),
            'target_review_rate':target.get('review_rate'),'best_round':run['best_round'],
            'trainable_parameters':run['trainable_encoder_parameters']+run['head_parameters']})
    with (folder/'development_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default='publication_20261017/scdd_development_v1')
    p.add_argument('--register-only',action='store_true');p.add_argument('--wait-for-calibration',action='store_true')
    p.add_argument('--maximum-new-runs',type=int);args=p.parse_args()
    folder=ROOT/'artifacts'/args.study;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registered=register(folder)
    print('REGISTERED',registered['design_id'],'180 real-domain development runs',flush=True)
    if args.register_only:return
    if args.wait_for_calibration:
        while not (CALIBRATION/'selected_hyperparameters.json').exists() or read_json(CALIBRATION/'status.json')['state']!='complete':
            status=read_json(CALIBRATION/'status.json')
            if status['state'] in {'failed','bounded_stop'}:raise RuntimeError('Calibration needs repair or resume: '+status['state'])
            save_json(folder/'status.json',{'state':'waiting_for_calibration','updated_at':now(),'pid':os.getpid(),
                'calibration_completed_runs':status['completed_runs'],'calibration_planned_runs':status['planned_runs']})
            time.sleep(30)
    settings=bind_hyperparameters(folder,registered);completed=0;new=0
    try:
        for fold in settings['split_manifest']['folds']:
            source_train,source_validation,target=fold_records(settings['split_manifest'],fold['name'])
            for budget in fold['budgets']:
                for seed in settings['seeds']:
                    for spec in settings['specifications']:
                        run_folder=folder/'runs'/fold['name']/f'budget_{budget:g}'/f"{spec['name']}_seed{seed}"
                        existed=(run_folder/'run.json').exists()
                        save_json(folder/'status.json',{'state':'running','updated_at':now(),'pid':os.getpid(),
                            'planned_runs':settings['planned_runs'],'completed_runs':completed,
                            'current':{'fold':fold['name'],'budget':budget,'model':spec['name'],'seed':seed}})
                        train(settings,spec,source_train,source_validation,seed,budget,run_folder)
                        if target:evaluate_development(settings,run_folder,target)
                        completed+=1;new+=int(not existed);summarize(folder)
                        if args.maximum_new_runs and new>=args.maximum_new_runs and completed<settings['planned_runs']:
                            save_json(folder/'status.json',{'state':'bounded_stop','updated_at':now(),
                                'completed_runs':completed,'planned_runs':settings['planned_runs']});return
        assert completed==settings['planned_runs']
        save_json(folder/'status.json',{'state':'complete','updated_at':now(),'pid':os.getpid(),
            'planned_runs':settings['planned_runs'],'completed_runs':completed,
            'next':'Review development evidence; freeze a separate confirmation protocol before opening any test images.'})
    except BaseException as e:
        save_json(folder/'status.json',{'state':'failed','updated_at':now(),'pid':os.getpid(),
            'completed_runs':completed,'planned_runs':settings['planned_runs'],'error':str(e),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
