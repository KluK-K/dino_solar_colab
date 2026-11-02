"""Complete original local-memory and post-LoRA frozen-bank controls, source-only."""
import argparse
import csv
import fcntl
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from elresearch.common import ROOT,read_json,save_json,digest,sha256,now,choose_device
from elresearch.data import records
from elresearch.publication_train import budget_rows
from elresearch.publication_splits import fold_records
from elresearch.mechanistic_train import FILES,scientific_hashes,train
from elresearch.normal_distance_fit import fit_normal_distance
from elresearch.mechanistic_evaluate import evaluate_mechanistic
from elresearch.postadapt_encoder import export_source_encoder


BASE=ROOT/'artifacts/publication_20261017'
DENSE=BASE/'dense_development_v1'
PARENT=BASE/'representation_scale_v1'


def registered_design():
    parent=read_json(DENSE/'protocol.json');split=parent['split_manifest']
    patch_runs=sum(3*len(f['budgets']) for f in split['folds'])
    distance_runs=sum(sum(3 if b<1 else 1 for b in f['budgets']) for f in split['folds'])
    adapted_parents=3+patch_runs
    settings={k:parent[k] for k in ['input_size','batch_size','updates_per_round','maximum_rounds','patience','weight_decay','target_source_recall','selection']}
    settings.update(version=1,kind='Explicit local-patch, normal-distance and frozen post-LoRA normal-reference mechanism controls',
        registered_after='Internal and first acquisition-holdout image-label development; before new SCDD dense results or any final confirmation scores',
        split_manifest=split,device='mps',seeds=[11,22,33],
        source_training_ids=[r['id'] for r in records('ucf','train')],
        source_validation_ids=[r['id'] for r in records('ucf','validation')],
        source_manifest_sha256=sha256(ROOT/'data/ucf/manifest.json'),
        upstream_dense_protocol={'path':str((DENSE/'protocol.json').relative_to(ROOT)),'sha256':sha256(DENSE/'protocol.json')},
        calibration={'dataset':'ucf','head_learning_rates':[3e-4,1e-3],'patch_fits':6,'distance_fits':1,
            'selection':'highest mean native UCF validation pixel AP over three seeds; earliest rate on ties'},
        local_patch={'training':'128-hidden MLP on frozen last-layer patch vectors;3-class patch targets max-preserve any annotated crack. Image BCE+weighted patch CE+0.5 patch Dice.',
            'localization':'Bilinear upsample of patch logits; no detail skip/multidepth decoder/normal references',
            'fairness':'Same original image/pixel annotation identities and update limit; patch objective differs from full-resolution dense loss and is reported explicitly.'},
        distance={'memory':'Same source-only normal eligibility/256-frame coverage/D4 and query-family exclusion as reference method; exact cosine NN within <=64 evenly indexed normal patches/frame/view, chunk2048',
            'operating_point':'99th percentile of known noncrack transformed response pixels in fixed source-validation samples; fixed scale10 sigmoid response. Not a crack posterior.',
            'image':'Top5% geometric-image cosine distance evidence; source labels choose high-recall threshold and temperature',
            'comparison':'Native pixel AP primary. Fixed learned-posterior0.5 versus source normal-FPR cutoff are different decision conventions; do not silently equate their Dice.',
            'seeds':'Only training subsets vary; full-budget deterministic memory saved once,seed11.'},
        postadapt={'parents':'All source-selected dense LoRA UCF models and predeclared SCDD source/fold/budget/seed LoRA models; no target selection',
            'export':'Load exact16 source LoRA Q/V matrices; merge into official architecture; verify token equivalence and fresh exported-weight reload on one source TRAINING image',
            'frozen_stage':'New identically initialized decoder heads, no-reference versus normal-reference. Freeze encoder; re-encode all references using actual merged-weight SHA in cache identity.',
            'learning_rate':'Use the already source-selected frozen dense head rate from UCF; no target/held-source hyperparameter search',
            'annotation_cost':'Same budgeted images/masks as parent. Report parent adaptation+fresh-head computation as two stages, not equal training cost.',
            'storage':'Exact canonical normal references persist. Rebuild hashed noncanonical D4 training intermediates on resume; remove only private derived adapted caches. APFS copy-on-write safetensors preserve complete merged models while changing only8 Q/V matrices.'},
        planned_local_patch_runs=6+patch_runs,planned_distance_runs=1+distance_runs,
        planned_postadapt_runs=2*adapted_parents,planned_runs=6+patch_runs+1+distance_runs+2*adapted_parents,
        limits=['DINOv2 disabled. No ELPV course test/SCDD117/UCF507/PVEL scoring.',
            'UCF574 labels select head rates and upstream LoRA rates; later SCDD-to-UCF is development-exposed transfer.',
            'Published normal-memory concepts are method controls, not an exact AnomalyDINO/SuperADD/PV-S3 reproduction.',
            'Patch and normal-distance models cannot resolve arbitrary subpatch detail; native metrics preserve that limitation.',
            'Retain failed hypotheses, all seeds and extra-stage computation; no first-DINOv3 or universal-superiority claim.'],
        scientific_hashes=scientific_hashes(),runner_sha256=sha256(__file__))
    settings['study_id']=digest(settings)[:20];return settings


def register(folder):
    settings=registered_design();path=folder/'protocol.json'
    if path.exists():
        old=read_json(path)
        if digest({k:v for k,v in old.items() if k!='registered_at'})!=digest(settings):
            raise RuntimeError('Registered mechanism-control protocol changed')
        return old
    snapshot=folder/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for name in FILES:shutil.copy2(ROOT/'elresearch'/name,snapshot/name)
    shutil.copy2(__file__,snapshot/Path(__file__).name)
    settings['registered_at']=now();save_json(path,settings);return settings


def check_binding(settings):
    if scientific_hashes()!=settings['scientific_hashes'] or sha256(__file__)!=settings['runner_sha256']:
        raise RuntimeError('Registered mechanism implementation changed')
    if sha256(ROOT/settings['upstream_dense_protocol']['path'])!=settings['upstream_dense_protocol']['sha256']:
        raise RuntimeError('Upstream dense source protocol changed')
    if sha256(ROOT/'data/ucf/manifest.json')!=settings['source_manifest_sha256']:
        raise RuntimeError('Source manifest changed')


def summarize(folder):
    rows=[]
    for path in sorted(folder.glob('**/run.json')):
        run=read_json(path);target=path.parent/'target_development.json'
        metric=read_json(target)['metrics'] if target.exists() else {}
        rows.append({'run_path':str(path.relative_to(ROOT)),'model':run['model']['name'],'seed':run['seed'],
            'training_fraction':run['training_budget_fraction'],'training_images':run['training_images'],
            'source_image_AUROC':run['validation_metrics']['auc'],
            'source_native_pixel_AP':run['validation_localization']['pooled_pixel_average_precision'],
            'target_image_AUROC':metric.get('auc'),'target_recall':metric.get('recall')})
    if rows:
        with (folder/'development_results.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--wait-for-representation-features',action='store_true');args=parser.parse_args()
    folder=BASE/'mechanistic_controls_v1';folder.mkdir(exist_ok=True)
    lock=(folder/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=register(folder);print('MECHANISM CONTROLS REGISTERED',settings['study_id'],settings['planned_runs'],'fits/calibrations',flush=True)
    if args.register_only:return
    completed=0
    def status(state,**extra):save_json(folder/'status.json',{'at':now(),'state':state,'pid':os.getpid(),
        'completed_runs':completed,'planned_runs':settings['planned_runs'],**extra})
    def execute(spec,tr,va,seed,budget,destination,fold,target=None):
        nonlocal completed
        check_binding(settings)
        if not (destination/'run.json').exists() and shutil.disk_usage(ROOT).free<3*1024**3:
            raise RuntimeError('Less than3GiB free before mechanism fit; preserve results and resolve recoverable cache/storage')
        status('running',current={'fold':fold,'budget':budget,'seed':seed,'model':spec['name']})
        fit=fit_normal_distance if spec['architecture']=='dinov3_normal_distance' else train
        result=fit(settings,spec,tr,va,seed,budget,destination)
        if target:evaluate_mechanistic({**settings,'permitted_target_development_ids':[r['id'] for r in target]},destination,target)
        completed+=1;summarize(folder);return result
    try:
        while True:
            prior=read_json(PARENT/'status.json')
            # These states are written only AFTER the parent releases its GPU lock.
            if prior['state'] in {'calibrating','running','complete'}:break
            if not args.wait_for_representation_features or prior['state'] in {'failed','bounded_stop'}:
                raise RuntimeError('Prior representation feature preparation needs completion/resume')
            status('queued_after_representation_features',parent_state=prior['state']);time.sleep(30)
        check_binding(settings)
        gpu_lock=(BASE/'gpu_work_queue.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
        tr,va=records('ucf','train'),records('ucf','validation');calibration=[]
        for rate in settings['calibration']['head_learning_rates']:
            spec={'name':f'dinov3_patch_mlp_lr{rate:g}','architecture':'dinov3_local_patch','adaptation':'frozen',
                'reference':False,'encoder_lr':0.,'head_lr':rate}
            for seed in settings['seeds']:
                result=execute(spec,tr,va,seed,1.,folder/'calibration'/f"{spec['name']}_seed{seed}",'ucf_patch_calibration')
                calibration.append(result)
        rate=max(settings['calibration']['head_learning_rates'],key=lambda rate:np.mean([
            r['validation_localization']['pooled_pixel_average_precision'] for r in calibration if r['model']['head_lr']==rate]))
        patch={'name':'dinov3_patch_mlp','architecture':'dinov3_local_patch','adaptation':'frozen','reference':False,'encoder_lr':0.,'head_lr':rate}
        distance={'name':'dinov3_global_normal_distance','architecture':'dinov3_normal_distance','adaptation':'frozen','reference':True,'encoder_lr':0.,'head_lr':0.}
        selection={'study_id':settings['study_id'],'protocol_sha256':sha256(folder/'protocol.json'),
            'selected_patch':patch,'rule':settings['calibration']['selection']}
        path=folder/'selected_hyperparameters.json'
        if path.exists() and read_json(path)!=selection:raise RuntimeError('Source patch hyperparameters changed')
        save_json(path,selection)
        execute(distance,tr,va,11,1.,folder/'source_transfer'/'ucf_distance_seed11','ucf_distance_calibration')
        for fold in settings['split_manifest']['folds']:
            source_train,source_val,target=fold_records(settings['split_manifest'],fold['name'])
            for budget in fold['budgets']:
                for spec in [patch,distance]:
                    for seed in ([11] if spec['architecture']=='dinov3_normal_distance' and budget==1 else settings['seeds']):
                        execute(spec,source_train,source_val,seed,budget,folder/'runs'/fold['name']/f'budget_{budget:g}'/f"{spec['name']}_seed{seed}",fold['name'],target)
        # Source-selected LoRA parents and frozen-head LR were fitted before this study.
        dense=read_json(DENSE/'protocol.json');selection=read_json(DENSE/'selected_hyperparameters.json')
        lora=next(x for x in selection['selected'] if x['adaptation']=='lora')
        frozen=next(x for x in selection['selected'] if x['adaptation']=='frozen')
        candidate=next(x for x in dense['calibration']['candidates'] if x['family']=='dinov3_dense_lora'
            and x['encoder_lr']==lora['encoder_lr'] and x['head_lr']==lora['head_lr'])
        parents=[]
        for seed in settings['seeds']:
            parents.append(('ucf_source',1.,seed,tr,va,[],DENSE/'calibration'/f"{candidate['name']}_seed{seed}"))
        for fold in settings['split_manifest']['folds']:
            source_train,source_val,target=fold_records(settings['split_manifest'],fold['name'])
            for budget in fold['budgets']:
                for seed in settings['seeds']:
                    parents.append((fold['name'],budget,seed,source_train,source_val,target,
                        DENSE/'runs'/fold['name']/f'budget_{budget:g}'/f'dinov3_dense_lora_seed{seed}'))
        for fold,budget,seed,source_train,source_val,target,parent in parents:
            selected=budget_rows(source_train,budget,seed)
            exported=folder/'adapted_encoders'/fold/f'budget_{budget:g}'/f'seed{seed}'
            status('exporting_adapted_encoder',current={'fold':fold,'budget':budget,'seed':seed})
            info=export_source_encoder(parent,selected,settings['input_size'],choose_device(settings['device']),exported)
            for reference in [False,True]:
                name='dinov3_postlora_reference' if reference else 'dinov3_postlora_no_reference'
                spec={'name':name,'architecture':'dinov3_postadapt','adaptation':'frozen','reference':reference,
                    'encoder_lr':0.,'head_lr':frozen['head_lr'],'adapted_encoder_path':str(exported.relative_to(ROOT)),
                    'adapted_weights_sha256':info['weights_sha256'],'adapted_config_sha256':info['config_sha256'],
                    'adaptation_parent_run':info['signature']['parent_run_path'],
                    'adaptation_parent_run_sha256':info['signature']['parent_run_sha256'],
                    'parent_hyperparameter_selection_sha256':sha256(DENSE/'selected_hyperparameters.json')}
                execute(spec,source_train,source_val,seed,budget,folder/'runs'/fold/f'budget_{budget:g}'/f'{name}_seed{seed}',fold,target)
        if completed!=settings['planned_runs']:raise RuntimeError('Mechanism fit count differs from original registered scope')
        status('complete',next='Compare matched source-only controls and mechanisms, retain failures; final confirmation remains separately sealed')
    except BaseException as error:
        status('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
