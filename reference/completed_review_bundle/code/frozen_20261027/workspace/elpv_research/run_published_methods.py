"""Full published architecture/mechanisms, source-calibrated canonical EL port."""
import argparse
import csv
import fcntl
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,digest,now
from elresearch.data import records
from elresearch.publication_splits import fold_records
from elresearch.published_train import FILES,scientific_hashes,train_published
from elresearch.published_archive import ensure_head_basis
from elresearch.published_weights import backbone_identity
from elresearch.published_evaluate import evaluate_published
from elresearch.publication_dense_train import train as train_dense,scientific_hashes as dense_hashes

BASE=ROOT/'artifacts/publication_20261017'
DENSE=BASE/'dense_development_v1'
PARENT=BASE/'mechanistic_controls_v1'


def calibration_candidates():
    candidates=[]
    for rate in [.001,.01]:
        for negative in [0.,.8]:
            candidates.append({'name':f'published_ssl_lr{rate:g}_neg{negative:g}',
                'semi':True,'learning_rate':rate,'vat_xi':.1,'negative_threshold':negative,
                'role':'Source-calibrated complete-mechanism FP32 canonical port'})
    for rate in [.001,.01]:
        candidates.append({'name':f'published_supervised_lr{rate:g}',
            'semi':False,'learning_rate':rate,'role':'Same full architecture; no teacher or unlabelled images'})
    return candidates


def registered_design():
    parent=read_json(DENSE/'protocol.json')
    settings={'version':1,'kind':'Full published PS-MT/PV-S3 architecture and SSL mechanisms; documented canonical3-class port',
        'registered_after':'Image-only development and implementation probes; before any new published-method performance result or independent confirmation score',
        'split_manifest':parent['split_manifest'],'seeds':[11,22,33],'input_size':336,'batch_size':2,'device':'mps',
        'minimum_updates_per_round':50,'maximum_rounds':40,'patience':6,'target_source_recall':.95,
        'update_plan':'max(50,ceil(all source-training images/2)) updates/round for both FSL and SSL; same step budget within each fold, no reduction of the published network',
        'source_training_ids':[r['id'] for r in records('ucf','train')],
        'source_validation_ids':[r['id'] for r in records('ucf','validation')],
        'source_manifest_sha256':sha256(ROOT/'data/ucf/manifest.json'),
        'upstream_dense_protocol':{'path':str((DENSE/'protocol.json').relative_to(ROOT)),'sha256':sha256(DENSE/'protocol.json')},
        'pretrained_initialization':backbone_identity(),
        'exact_archive_head_bases':[ensure_head_basis(seed,336) for seed in [11,22,33]],
        'published_source':{'PS_MT_commit':'058cdc3bd8fe5faa83dbc798cfb4c136b74207fd',
            'PV_S3_commit':'e062f81d602aea4cbae45c13ec85cfe79f95aa7e',
            'license':'MIT PS-MT architecture; do not redistribute PV-S3 code under an unverified license',
            'architecture_audit_sha256':sha256(BASE/'published_architecture_source.json')},
        'calibration':{'dataset':'ucf','candidates':calibration_candidates(),'planned_runs':18,
            'SSL_training_fraction':.2,'FSL_training_fraction':1.,
            'selection':'Highest mean native source-validation pixel AP across3 seeds; candidate order breaks ties. Select supervised and semi-supervised settings separately.',
            'literal_recipe_sensitivity':{'runs':3,'training_fraction':.2,'learning_rate':.01,'vat_xi':1e-6,'negative_threshold':0.,
                'role':'Released-caller numerical setting, fixed in advance; report source results separately and never use target scores to choose xi.'}},
        'numerical_VAT':{'primary_xi':.1,'epsilon':2.,'basis':'Released helper default and336px initial FP32 local precision diagnostic, not target scores',
            'diagnostic':{'path':str((BASE/'published_vat_precision_probe_336.json').relative_to(ROOT)),
                          'sha256':sha256(BASE/'published_vat_precision_probe_336.json')},
            'limit':'Initial source FP32 numerical behavior does not establish failure of the authors trained implementation on other hardware.'},
        'method':{'architecture':'Complete deep-stem dilated ResNet50/ASPP/low-level fusion/decoder;41,762,595 parameters per network, including source-defined unused layers',
            'SSL':'Random-initialized two frozen-gradient teachers and ImageNet-initialized student; decoder VAT on labelled/unlabelled student features; weak/strong photometry; assistance top(C-1); CutMix Beta4,4; confidence CE; alternating per-round EMA.996 including BN buffers.',
            'optimizer':'SGD(momentum.9,weight_decay1e-4),poly power.9; unsupervised maximum1.5 with12-round Gaussian ramp; no image BCE',
            'source_selection':'Fixed sampled source pixel AP. Primary inference is teacher1, matching released PV-S3 inference.py.',
            'extra_inference':'Predeclared mean of teacher logits at the exact same selected checkpoint, reflecting PS-MT paper inference; own source threshold/temperature; not another fit or target-selected view',
            'archive':'Lossless full-state byte-XOR/byte-shuffle/gzip against exact ImageNet+persisted head basis, including every floating BN buffer and integer counter; fresh prediction equality checked for each view.'},
        'supervision':{'labelled':'Canonical normal/crack/other masks on identical budgeted original source images; geometry IGNORE retained. Image labels define stratified retrospective subsets only.',
            'unlabelled':'Remaining source-TRAIN images only, without reading their masks/labels in the SSL loader. No held-source/validation/confirmation image in this pool.',
            'cost':'Training mask budgets and144 UCF validation masks are explicit. Full UCF calibration and upstream dense HPO use574 labelled images; do not claim total annotation cost equals the training percentage.',
            'DINO_matched_source_controls':'Frozen dense/no-reference, LoRA dense/no-reference, and frozen dense/normal-reference at UCF20%,3seeds; same86 training masks+144 source-validation masks; existing full-source-selected hyperparameters; same labels but different objectives/update counts.'},
        'development_design':{'folds':'Internal and ARTS/CFVS/CSIR held-acquisition-source development, same preexisting immutable fold IDs',
            'FSL_training_fractions':[.2,1.],'SSL_training_fractions':[.2],
            'FSL_fits':24,'SSL_fits':12,'matched_UCF_FSL20_fits':3,'matched_UCF_DINO20_fits':9},
        'adaptations':['Canonical3 classes replace published5; topk4 becomes top(C-1)=2. Ignore padding is geometry-masked.',
            'Whole-field336px letterboxing and D4 replace released224px crops; fixed source masks remain aligned.',
            'Local batch2 on MPS replaces released batch16/CUDA/distributed settings. Epoch/update and early stopping differ; record actual optimizer work.',
            'Gray replication makes source gray transform a no-op; reviewed brightness/contrast and Gaussian kernels retained.',
            'DeepLab source branch does not apply PSP-only Zoom; no silent claim that the different PSP branch was reproduced.',
            'Integer BN counters copied in EMA; all floating BN state updated/saved. VAT uses non-inplace features and restored decoder modes.',
            'Primary xi.1 numerical port and literalxi1e-6 source sensitivity are both explicit. This is not the published experiment or a reproduction of reported mIoU.'],
        'limits':['DINOv2 disabled. Final SCDD117/UCF507/PVEL scores remain unopened.',
            'UCF and SCDD source-exposure differs per model; label/HPO/UL image use must all remain visible.',
            'Teacher1 and ensemble share training; they are correlated inference views, not independent seeds.',
            'Published numerical scores are literature context; direct quantitative ranking requires same task, cohort and supervision.'],
        'planned_runs':69,'source_calibration_and_sensitivity_runs':21,
        'scientific_hashes':scientific_hashes(),'dense_control_scientific_hashes':dense_hashes(),
        'runner_sha256':sha256(__file__)}
    counts=settings['development_design']
    if len(settings['split_manifest']['folds'])!=4 or (len(calibration_candidates())*3+3+
        counts['FSL_fits']+counts['SSL_fits']+counts['matched_UCF_FSL20_fits']+counts['matched_UCF_DINO20_fits'])!=settings['planned_runs']:
        raise RuntimeError('Published-method schedule does not cover its declared complete scope')
    settings['study_id']=digest(settings)[:20];return settings


def register(folder):
    settings=registered_design();path=folder/'protocol.json'
    if path.exists():
        old=read_json(path)
        if digest({k:v for k,v in old.items() if k!='registered_at'})!=digest(settings):
            raise RuntimeError('Published-method registered protocol changed')
        return old
    snapshot=folder/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for name in FILES:shutil.copy2(ROOT/'elresearch'/name,snapshot/name)
    shutil.copy2(__file__,snapshot/Path(__file__).name)
    shutil.copy2(ROOT/'elresearch/licenses/PSMT_MIT_LICENSE.txt',snapshot/'PSMT_MIT_LICENSE.txt')
    settings['registered_at']=now();save_json(path,settings);return settings


def check_binding(settings):
    if scientific_hashes()!=settings['scientific_hashes'] or dense_hashes()!=settings['dense_control_scientific_hashes'] or sha256(__file__)!=settings['runner_sha256']:
        raise RuntimeError('Published-method registered implementation changed')
    binding=settings['upstream_dense_protocol']
    if sha256(ROOT/binding['path'])!=binding['sha256']:raise RuntimeError('Upstream dense protocol changed')
    if sha256(ROOT/'data/ucf/manifest.json')!=settings['source_manifest_sha256']:raise RuntimeError('UCF source manifest changed')
    if backbone_identity()!=settings['pretrained_initialization']:raise RuntimeError('Published backbone identity changed')
    for head in settings['exact_archive_head_bases']:
        if sha256(ROOT/head['path'])!=head['sha256']:raise RuntimeError('Exact head archive basis changed')


def summarize(folder):
    rows=[]
    for path in sorted(folder.glob('**/run.json')):
        run=read_json(path);target=path.parent/'target_development.json';targets=read_json(target).get('inference_views',{}) if target.exists() else {}
        for view,source in run.get('inference_views',{'primary':run}).items():
            rows.append({'run_path':str(path.relative_to(ROOT)),'model':run['model']['name'],'seed':run['seed'],
                'training_fraction':run['training_budget_fraction'],'training_images':run['training_images'],
                'unlabelled_source_images':run.get('unlabelled_source_images',0),'inference_view':view,
                'source_image_AUROC':source['validation_metrics']['auc'],
                'source_pixel_AP':source['validation_localization']['pooled_pixel_average_precision'],
                'target_image_AUROC':targets.get(view,{}).get('metrics',{}).get('auc')})
    if rows:
        with (folder/'development_results.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--wait-for-mechanistic-study',action='store_true');args=parser.parse_args()
    folder=BASE/'published_methods_v1';folder.mkdir(exist_ok=True)
    lock=(folder/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=register(folder);print('PUBLISHED METHODS REGISTERED',settings['study_id'],settings['planned_runs'],'fits',flush=True)
    if args.register_only:return
    completed=0
    def status(state,**extra):save_json(folder/'status.json',{'at':now(),'state':state,'pid':os.getpid(),'completed_runs':completed,'planned_runs':settings['planned_runs'],**extra})
    def execute(spec,tr,va,seed,budget,destination,fold,target=None,dino=False):
        nonlocal completed
        check_binding(settings)
        if not (destination/'run.json').exists() and shutil.disk_usage(ROOT).free<3*1024**3:
            raise RuntimeError('Less than3GiB free before complete-model fit; preserve results and resolve recoverable storage')
        status('running',current={'fold':fold,'budget':budget,'seed':seed,'model':spec['name']})
        child={**settings,'scientific_hashes':settings['dense_control_scientific_hashes'],
            'updates_per_round':50,'weight_decay':read_json(DENSE/'protocol.json')['weight_decay']} if dino else settings
        result=train_dense(child,spec,tr,va,seed,budget,destination) if dino else train_published(child,spec,tr,va,seed,budget,destination)
        if target:
            evaluate_published({**settings,'permitted_target_development_ids':[r['id'] for r in target]},destination,target)
        completed+=1;summarize(folder);return result
    try:
        while True:
            prior=read_json(PARENT/'status.json')
            if prior['state']=='complete':break
            if not args.wait_for_mechanistic_study or prior['state'] in {'failed','bounded_stop'}:
                raise RuntimeError('Prior mechanism controls need completion/resume')
            status('queued_after_mechanistic_study',parent_state=prior['state']);time.sleep(30)
        gpu=(BASE/'gpu_work_queue.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX)
        check_binding(settings);tr,va=records('ucf','train'),records('ucf','validation');calibration=[]
        for spec in settings['calibration']['candidates']:
            for seed in settings['seeds']:
                calibration.append(execute(spec,tr,va,seed,.2 if spec['semi'] else 1.,
                    folder/'calibration'/f"{spec['name']}_seed{seed}",'ucf_source_calibration'))
        selected=[]
        for semi in [False,True]:
            candidates=[s for s in settings['calibration']['candidates'] if s['semi']==semi]
            best=max(candidates,key=lambda s:np.mean([r['validation_localization']['pooled_pixel_average_precision']
                for r in calibration if r['model']['name']==s['name']]))
            selected.append({**best,'name':'published_psmt_ssl' if semi else 'published_deeplab_fsl'})
        selection={'study_id':settings['study_id'],'source_protocol_sha256':sha256(folder/'protocol.json'),
            'selected':selected,'rule':settings['calibration']['selection']}
        path=folder/'selected_hyperparameters.json'
        if path.exists() and read_json(path)!=selection:raise RuntimeError('Source-selected published hyperparameters changed')
        save_json(path,selection)
        literal={'name':'published_ssl_literal_xi1e6','semi':True,'learning_rate':.01,'vat_xi':1e-6,'negative_threshold':0.,'role':'Fixed released-caller source-only sensitivity'}
        for seed in settings['seeds']:
            execute(literal,tr,va,seed,.2,folder/'source_sensitivity'/f"{literal['name']}_seed{seed}",'ucf_literal_sensitivity')
        for fold in settings['split_manifest']['folds']:
            source_train,source_val,target=fold_records(settings['split_manifest'],fold['name'])
            for spec in selected:
                for budget in ([.2] if spec['semi'] else [.2,1.]):
                    for seed in settings['seeds']:
                        execute(spec,source_train,source_val,seed,budget,folder/'runs'/fold['name']/f'budget_{budget:g}'/f"{spec['name']}_seed{seed}",fold['name'],target)
        for seed in settings['seeds']:
            execute(selected[0],tr,va,seed,.2,folder/'source_transfer'/f'published_deeplab_fsl_seed{seed}','ucf_matched20')
        upstream=read_json(DENSE/'selected_hyperparameters.json')['selected']
        frozen=next(x for x in upstream if x['adaptation']=='frozen')
        lora=next(x for x in upstream if x['adaptation']=='lora')
        controls=[{**frozen,'name':'dinov3_frozen_ucf20'},
            {**lora,'name':'dinov3_lora_ucf20'},
            {**frozen,'name':'dinov3_reference_ucf20','reference':True,'geometry_weight':.1,'top_fraction':.05}]
        control_binding={'source_dense_selection_sha256':sha256(DENSE/'selected_hyperparameters.json'),'selected':controls,
            'annotation_identity_rule':'Same budget_rows(source,.2,seed) and144 source-validation masks as published FSL/SSL; DINO optimizer/loss/update limits remain its registered original settings.'}
        save_json(folder/'matched_dino_controls.json',control_binding)
        for spec in controls:
            for seed in settings['seeds']:
                execute(spec,tr,va,seed,.2,folder/'source_transfer'/f"{spec['name']}_seed{seed}",'ucf_matched20',dino=True)
        if completed!=settings['planned_runs']:raise RuntimeError('Published comparison fit count differs from registered scope')
        status('complete',next='Retain every seed and published inference view, report adaptations and compute; final confirmation remains separately sealed')
    except BaseException as error:
        status('failed',error=str(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
