"""Explicit source-reviewed confirmation preparation, execution and output audit.

This module never chooses a winning model from confirmation scores. Preparing a
plan requires all source studies and the complete efficiency evidence. A reviewed
specification supplies concrete source-model declarations and comparisons.
"""
from pathlib import Path
import gc
import math
import fcntl
import os
import shutil
import traceback

import numpy as np

from .common import ROOT,read_json,save_json,sha256,digest,now
from . import confirmation_registry as gate
from .confirmation_evaluate import evaluate
from .metrics import classification
from .publication_train import probability_metrics

BASE=ROOT/'artifacts/publication_20261017'
FINAL=BASE/'final_confirmation_v1'
STUDIES={'image':'scdd_development_v1','classical':'classical_controls_v1','dense':'dense_development_v1',
    'ucf_reference_transfer':'ucf_reference_transfer_v1','cnn_stability':'cnn_stability_v1',
    'representation_scale':'representation_scale_v1','mechanistic':'mechanistic_controls_v1','published_methods':'published_methods_v1'}
MODEL_FIELDS={'kind','family','seed','run_folder','source_protocol','source_view','cohorts','analysis_stratum','analysis_role'}


def local_path(value):
    path=(ROOT/value).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise PermissionError('Confirmation inputs must stay in the research workspace')
    return path


def development_evidence():
    result={}
    for name,study in STUDIES.items():
        folder=BASE/study;protocol=read_json(folder/'protocol.json')
        expected=(protocol['planned_real_domain_runs']+protocol['calibration']['planned_runs'] if name=='dense' else
            protocol['development_logical_runs'] if name=='classical' else protocol['planned_runs'])
        pattern=('runs/*/*/*/run.json' if name in {'image','classical'} else
                 'runs/*/run.json' if name in {'ucf_reference_transfer','cnn_stability'} else '**/run.json')
        result[name]={'status':gate.bind_file(folder/'status.json'),'protocol':gate.bind_file(folder/'protocol.json'),
            'expected_runs':expected,'completed_run_glob':pattern}
    return result


def compile_jobs(declarations,models):
    if len(declarations)!=len(models):raise ValueError('Model declarations/bindings do not align')
    jobs=[];seen=set()
    for declaration,model in zip(declarations,models):
        cohorts=declaration.get('cohorts',[])
        if not cohorts or len(set(cohorts))!=len(cohorts):raise ValueError('Each source model needs unique declared target cohorts')
        for cohort in cohorts:
            if cohort not in gate.ALLOWED_COHORTS:raise PermissionError('Unknown confirmation cohort')
            key=(model['id'],cohort)
            if key in seen:raise ValueError('Repeated model/cohort evaluation job')
            seen.add(key);stratum=declaration.get('analysis_stratum','whole')
            if stratum!='whole' and (cohort!='scdd_publication_v1' or stratum not in {'ARTS','CFVS','CSIR'}):
                raise ValueError('Source-held-out strata refer only to audited SCDD collection channels')
            role=declaration.get('analysis_role','')
            if not role.strip():raise ValueError('Declare the transfer/in-dataset interpretation for every model/cohort')
            jobs.append({'model_id':model['id'],'cohort':cohort,'analysis_stratum':stratum,'analysis_role':role})
    return jobs


def propose_fixed_examples(rows):
    """Balanced by provided acquisition channel/class, ordered only by file ID."""
    groups={}
    for row in rows:
        if row['target'] not in {0,1}:raise ValueError('Unknown image truth cannot define a fixed gallery')
        key=(str(row.get('acquisition_source') or 'unspecified'),row['target'])
        groups.setdefault(key,[]).append(row)
    selected=[]
    for key,values in sorted(groups.items()):
        # Evenly spaced IDs are a deterministic metadata sample, not physical
        # groups and not examples chosen to make a model look good or bad.
        values=sorted(values,key=lambda r:r['id']);count=1 if len(groups)>=6 else 3
        positions=sorted(set(np.linspace(0,len(values)-1,min(count,len(values)),dtype=int).tolist()))
        selected.extend(values[i]['id'] for i in positions)
    if len(selected)>12:raise ValueError('Declare a bounded scientific gallery policy for this many channels')
    return selected


def validate_fixed_examples(examples,models,jobs,cohorts,population_policy=None):
    spatial={j['cohort'] for j in jobs if next(m for m in models if m['id']==j['model_id'])['kind']=='dense'}
    for cohort in cohorts:
        if cohort['name'] not in spatial:continue
        rows=[r for r in read_json(gate.check_file(cohort['manifest']))['records'] if r['partition']==cohort['partition']]
        from .confirmation_population import selected_rows
        rows,_=selected_rows(rows,cohort['name'],population_policy)
        identifiers=examples.get(cohort['name'],[]);by_id={r['id']:r for r in rows}
        if not 2<=len(identifiers)<=12 or len(identifiers)!=len(set(identifiers)) or not set(identifiers)<=set(by_id):
            raise ValueError('Predeclare2–12 unique authorized native gallery identities for every spatial cohort')
        if {by_id[i]['target'] for i in identifiers}!={r['target'] for r in rows}:
            raise ValueError('Fixed native gallery must retain both annotated-positive and negative examples when available')


def specification_template():
    return {'version':1,'source_review_completed':False,
        'source_review':{'rationale':'','inputs':[]},'models':[],
        'primary_comparisons':[],'primary_endpoints':[],
        'uncertainty':{'scope':'Conditional paired acquisition-family estimates; no patch pseudoreplication or invented PVEL module intervals.',
            'statistics_plan':'artifacts/publication_20261017/statistics_plan_v2.json'},
        'fixed_localization_examples':{},'inference_device':'mps',
        'cohort_analysis_policy':'artifacts/publication_20261017/confirmation_preparation_v1/cohort_analysis_policy_v1.json',
        'secondary_native_failure_policy':'artifacts/publication_20261017/confirmation_preparation_v1/secondary_native_failure_policy_v1.json',
        'storage':{'minimum_free_GiB':2.,'retire_verified_native_pixel_cache':True},
        'limits':['UCF source provenance conditional; target dataset/HPO/UL exposure reported per model.',
                  'The research design has observed UCF and SCDD development outcomes; a source-only fitted model does not make either dataset wholly unseen to the research process.',
                  'Different pretrained corpora do not identify an architecture-only effect.',
                  'PVEL positives are explicitly annotated crack/star; negatives author-normal only. Other-defect specificity unknown.',
                  'Held-source model strata must be reported separately; no unexplained AUROC pooled across different models.']}


def validate_primary_comparisons(comparisons,models,jobs):
    """Concrete paired families/endpoints, with explicit annotation matching."""
    by_family={}
    for model in models:by_family.setdefault(model['family'],[]).append(model)
    image_endpoints={'AUROC','average_precision','recall_at_source_threshold','review_rate_at_source_threshold','false_positive_rate_at_source_threshold'}
    pixel_endpoints={'pooled_native_pixel_AP','positive_image_Dice','positive_image_IoU','normal_structure_pixel_FPR'}
    names=set()
    for contrast in comparisons:
        required={'id','first_family','second_family','cohort','endpoint','analysis_stratum','matched_annotation_identities'}
        if not isinstance(contrast,dict) or not required<=set(contrast):raise ValueError('Primary comparisons require concrete family/cohort/endpoint declarations')
        if contrast['id'] in names:raise ValueError('Duplicate primary comparison identifier')
        names.add(contrast['id'])
        first,second=contrast['first_family'],contrast['second_family'];cohort=contrast['cohort']
        if first==second or first not in by_family or second not in by_family:raise ValueError('Primary comparison families missing or identical')
        if cohort not in gate.ALLOWED_COHORTS:raise PermissionError('Unknown primary cohort')
        allowed={'known_crack_recall','author_normal_FPR','box_point_hit_rate'} if cohort.startswith('pvelad') else image_endpoints|pixel_endpoints
        if contrast['endpoint'] not in allowed:raise ValueError('Endpoint does not match cohort annotation/task scope')
        if contrast['endpoint']=='normal_structure_pixel_FPR' and cohort!='scdd_publication_v1':
            raise ValueError('Only SCDD supplies the required normal-structure annotation')
        stratum=contrast['analysis_stratum']
        if stratum!='whole' and (cohort!='scdd_publication_v1' or stratum not in {'ARTS','CFVS','CSIR'}):
            raise ValueError('Unknown primary acquisition-source stratum')
        if contrast['endpoint']=='pooled_native_pixel_AP' and stratum!='whole':
            raise ValueError('Pooled native AP is stored for the whole scored cohort; use positive-image Dice for a grouped stratum')
        if not isinstance(contrast['matched_annotation_identities'],bool):raise ValueError('Explicitly declare whether source annotation identities match')
        def deterministic(values):
            return len(values)==1 and values[0]['training_fraction']==1. and (values[0]['kind']=='classical' or values[0].get('architecture')=='dinov3_normal_distance')
        first_models,second_models=by_family[first],by_family[second]
        seeds_first={m['seed']:m for m in first_models};seeds_second={m['seed']:m for m in second_models}
        if set(seeds_first)!=set(seeds_second) and not (deterministic(first_models) or deterministic(second_models)):
            raise ValueError('Primary comparison lacks paired stochastic/subset seeds')
        if contrast['endpoint'] in pixel_endpoints or contrast['endpoint']=='box_point_hit_rate':
            if any(m['kind']!='dense' for m in first_models+second_models):raise ValueError('Localization endpoint requires spatial-output models on both sides')
        for model in first_models+second_models:
            applicable=[j for j in jobs if j['model_id']==model['id'] and j['cohort']==cohort]
            if len(applicable)!=1:raise ValueError('Primary comparison has no unique frozen evaluation job')
            if applicable[0].get('analysis_stratum','whole') not in {'whole',stratum}:
                raise ValueError('Primary comparison ignores the declared matching held-source stratum')
        if contrast['matched_annotation_identities']:
            for seed in set(seeds_first)|set(seeds_second):
                a=seeds_first.get(seed,first_models[0]);b=seeds_second.get(seed,second_models[0])
                if set(a['source_train_ids'])!=set(b['source_train_ids']) or set(a['source_validation_ids'])!=set(b['source_validation_ids']):
                    raise ValueError('Claimed matched annotation identities differ')


def prepare(specification_path,destination):
    if (FINAL/'frozen_registry.json').exists() or (FINAL/'observation_opened.json').exists():
        raise PermissionError('A real confirmation registry already exists; only its exact evaluation continuation is allowed')
    specification_path=local_path(specification_path);destination=local_path(destination);spec=read_json(specification_path)
    if spec.get('source_review_completed') is not True or not spec.get('source_review',{}).get('rationale','').strip():
        raise RuntimeError('Complete the scientific source-evidence review; a template is not a confirmation decision')
    if not spec.get('primary_comparisons') or not spec.get('primary_endpoints') or not spec.get('models'):
        raise ValueError('Concrete primary comparisons, endpoints and source models are required')
    evidence=development_evidence();gate.validate_development_evidence(evidence)
    efficiency=gate.verified_inference_efficiency()
    from .elpv_supplement import verify_complete as verify_elpv_supplement
    supplementary=BASE/'elpv_review_supplement_v1'
    supplementary_proof=verify_elpv_supplement(supplementary)
    review=[]
    for value in spec['source_review']['inputs']:
        path=local_path(value)
        if path.is_relative_to(FINAL):raise PermissionError('Confirmation outcomes cannot be source-review/model-selection inputs')
        review.append(gate.bind_file(path))
    if not review:raise ValueError('Bind the actual source-evidence files supporting the review')
    models=[]
    for declaration in spec['models']:
        if set(declaration)-MODEL_FIELDS:raise ValueError('Unsupported model overrides, including target threshold/scaler fitting')
        models.append(gate.bind_model(declaration['kind'],declaration['family'],declaration['seed'],
            local_path(declaration['run_folder']),local_path(declaration['source_protocol']),declaration.get('source_view','primary')))
    if len({m['id'] for m in models})!=len(models):raise ValueError('Unique family/seed/view model identities required')
    jobs=compile_jobs(spec['models'],models);names=list(dict.fromkeys(j['cohort'] for j in jobs))
    validate_primary_comparisons(spec['primary_comparisons'],models,jobs)
    cohorts=[gate.cohort_binding(name) for name in names]
    from .confirmation_population import checked_policy
    if not spec.get('cohort_analysis_policy'):raise ValueError('Bind the pre-inference near-copy population policy, including full-pool sensitivity')
    population_binding=gate.bind_file(local_path(spec['cohort_analysis_policy']));population=checked_policy(population_binding)
    from .secondary_failure_analysis import checked_failure_policy
    if not spec.get('secondary_native_failure_policy'):
        raise ValueError('Bind the source-fixed secondary annotation-footprint/zero-overlap policy before final scoring')
    failure_binding=gate.bind_file(local_path(spec['secondary_native_failure_policy']))
    failure_policy=checked_failure_policy(failure_binding)
    for comparison in spec['primary_comparisons']:
        exclusions=population['cohorts'].get(comparison['cohort'],{}).get('subsets',{}).get('primary',{}).get('excluded_ids',[])
        if exclusions and comparison['endpoint']=='pooled_native_pixel_AP':
            raise ValueError('Do not substitute an unfiltered pooled pixel AP for a filtered primary cohort')
    validate_fixed_examples(spec['fixed_localization_examples'],models,jobs,cohorts,population)
    files=[ROOT/'elresearch/confirmation_workflow.py',ROOT/'run_confirmation_workflow.py',
        ROOT/'elresearch/publication_statistics.py',ROOT/'elresearch/inspection_scenarios.py',
        ROOT/'elresearch/confirmation_analysis.py',ROOT/'elresearch/elpv_supplement.py',ROOT/'run_elpv_supplement.py',
        ROOT/'elresearch/confirmation_population.py',gate.check_file(population_binding),
        ROOT/'elresearch/secondary_failure_analysis.py',gate.check_file(failure_binding),
        ROOT/'elresearch/near_duplicate_audit.py',ROOT/'run_near_duplicate_audit.py',ROOT/'review_similarity_candidates.py',
        supplementary/'protocol.json',supplementary/'results.json']
    files.extend(gate.check_file(value) for value in population['evidence'])
    files.extend(gate.check_file(value) for value in failure_policy['evidence'].values())
    source_failure_plan=read_json(gate.check_file(failure_policy['evidence']['source_plan']))
    for row in source_failure_plan['source_bindings']:
        run_path=gate.check_file({'path':row['run'],'sha256':row['sha256']})
        prediction_path=run_path.parent/'validation_predictions.npz'
        gate.check_file({'path':str(prediction_path.relative_to(ROOT)),'sha256':row['prediction_sha256']})
        files.extend([run_path,prediction_path])
    statistics_path=local_path(spec['uncertainty']['statistics_plan'])
    statistics=read_json(statistics_path)
    if statistics['repetitions']!=1000 or statistics['seed']!=20261017:
        raise ValueError('Preserve the prespecified acquisition-family statistics')
    files.append(statistics_path)
    policy={'primary_comparisons':spec['primary_comparisons'],'primary_endpoints':spec['primary_endpoints'],
        'selection_rationale':spec['source_review']['rationale'],'source_review_inputs':review,
        'uncertainty':spec['uncertainty'],'statistics_plan_binding':gate.bind_file(statistics_path),
        'cohort_analysis_policy_binding':population_binding,
        'secondary_native_failure_policy_binding':failure_binding,
        'limits':spec['limits'],'inference_device':spec['inference_device'],
        'study_wide_design_context':{'observed_development_datasets':['ucf','scdd'],
            'interpretation':'Per-model gradient/checkpoint/hyperparameter exposure remains separate from study-wide method-design exposure. UCF-only weights provide source-fitted transfer to SCDD, whose development outcomes have also informed this research. PVEL model outcomes remain unopened until this registry; raw integrity/ontology audits are already observed.'},
        'confirmation_jobs':jobs,'fixed_localization_examples':spec['fixed_localization_examples'],
        'storage_policy':spec['storage'],'specification':gate.bind_file(specification_path),
        'supplementary_ELPV_review_CV':{'scope':'Different review-label task; not a crack-confirmation cohort or main-model training source.',
            'verification':supplementary_proof,'protocol':gate.bind_file(supplementary/'protocol.json'),
            'results':gate.bind_file(supplementary/'results.json')},
        'stage':'Separate immutable confirmation after source review; no target-fitted choices'}
    plan={'version':1,'prepared_at':now(),'models':models,'cohorts':cohorts,'development_evidence':evidence,
        'analysis_plan':policy,'evaluator_files':[gate.bind_file(path) for path in files],
        'source_identity_index':gate.bind_file(BASE/'source_development_identity_v1.json'),
        'verified_efficiency':efficiency}
    plan['preparation_id']=digest(plan)
    if destination.exists():
        old=checked_preparation(destination)
        comparable=lambda value:{k:v for k,v in value.items() if k not in {'prepared_at','preparation_id'}}
        if comparable(old)!=comparable(plan):raise RuntimeError('Prepared plan already exists; preserve it and use a new source-reviewed version')
        return old
    save_json(destination,plan);return plan


def checked_preparation(path):
    path=Path(path);plan=read_json(path)
    if digest({k:v for k,v in plan.items() if k!='preparation_id'})!=plan['preparation_id']:
        raise RuntimeError('Prepared confirmation plan changed')
    gate.check_file(plan['analysis_plan']['specification'])
    for binding in plan['analysis_plan']['source_review_inputs']+plan['evaluator_files']:gate.check_file(binding)
    return plan


def freeze_preparation(path):
    plan=checked_preparation(path)
    analysis={**plan['analysis_plan'],'prepared_plan':gate.bind_file(path),'preparation_id':plan['preparation_id']}
    return gate.freeze_registry(FINAL,plan['models'],plan['cohorts'],plan['development_evidence'],analysis,
        [gate.check_file(b) for b in plan['evaluator_files']]+[Path(path)],plan['source_identity_index'])


def verify_prediction_result(folder,registry,job):
    model=next(m for m in registry['models'] if m['id']==job['model_id'])
    cohort=next(c for c in registry['cohorts'] if c['name']==job['cohort'])
    destination=Path(folder)/'results'/model['id']/cohort['name'];result=read_json(destination/'result.json')
    if (result['registry_id']!=registry['registry_id'] or result['model_id']!=model['id'] or result['cohort']!=cohort['name']
        or result['predictions_sha256']!=sha256(destination/'predictions.npz')):
        raise RuntimeError('Confirmation prediction identity changed')
    manifest=read_json(gate.check_file(cohort['manifest']));rows=[r for r in manifest['records'] if r['partition']==cohort['partition']]
    with np.load(destination/'predictions.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['ids'],[r['id'] for r in rows]);np.testing.assert_array_equal(f['y'],[r['target'] for r in rows])
        actual=classification(f['y'],f['logits'],model['source_threshold'])
        calibrated=probability_metrics(f['y'],f['logits'],model['source_temperature'])
    if actual!=result['metrics'] or result['source_frozen_threshold']!=model['source_threshold'] or result['threshold_fitted_on_target']:
        raise RuntimeError('Reported confirmation decisions do not match saved frozen predictions')
    if result['source_frozen_temperature']!=model['source_temperature'] or result['source_temperature_probability_metrics']!=calibrated:
        raise RuntimeError('Reported confirmation calibration differs from frozen source temperature')
    return destination,result,rows


def retire_native_cache(folder,registry,job):
    """Retire only owned reproducible arrays AFTER complete checked metrics exist."""
    destination,result,rows=verify_prediction_result(folder,registry,job)
    cache=destination/'native_pixel_cache';marker=destination/'native_cache_retirement.json'
    if cache.exists() and (cache.is_symlink() or cache.resolve().parent!=destination.resolve()):raise PermissionError('Unsafe native cache location')
    local=result.get('localization') or {};metrics={r['id']:r for r in local.get('native_pixel_per_image',[])}
    expected={digest(row['id']):row for row in rows if row['id'] in metrics}
    if marker.exists():
        old=read_json(marker)
        if old['registry_id']!=registry['registry_id'] or old['result_sha256']!=sha256(destination/'result.json'):
            raise RuntimeError('Native cache retirement belongs to different results')
        if {Path(item['file']).stem for item in old['files']}!=set(expected):raise RuntimeError('Retired cache ownership changed')
        for item in old['files']:
            name=item['file']
            if Path(name).name!=name or Path(name).suffix!='.npz':raise PermissionError('Unsafe retired cache name')
            path=cache/name
            if path.exists() and (path.is_symlink() or sha256(path)!=item['sha256']):raise RuntimeError('Pending retired cache changed')
        if old['status']=='retirement_prepared':
            for item in old['files']:(cache/item['file']).unlink(missing_ok=True)
            old['status']='retired';save_json(marker,old)
        elif old['status']!='retired' or list(cache.glob('*.npz')):
            raise RuntimeError('Unexpected cache files/state after recorded retirement')
        return old
    if not cache.exists():return None
    if cache.is_symlink() or cache.resolve().parent!=destination.resolve():raise PermissionError('Unsafe native cache location')
    proof=[]
    files=list(cache.glob('*.npz'))
    if {path.stem for path in files}!=set(expected):raise RuntimeError('Native cache identities differ from fully scored pixel cohort')
    # Validate all files first; no partial pruning on a mismatched cache.
    for path in files:
        if path.is_symlink() or path.resolve().parent!=cache.resolve():raise PermissionError('Unsafe derived cache file')
        row=expected[path.stem];metric=metrics[row['id']]
        with np.load(path,allow_pickle=False) as value:
            truth=value['truth'];score=value['score']
            if (truth.dtype!=np.bool_ or score.dtype!=np.float32 or truth.ndim!=1 or truth.shape!=score.shape or len(truth)!=metric['known_pixels']
                or int(truth.sum())!=metric['crack_pixels'] or not np.isfinite(score).all()
                or (len(score) and (score.min()<0 or score.max()>1))):
                raise RuntimeError('Derived pixel cache does not match completed native metrics')
        proof.append({'id':row['id'],'file':path.name,'sha256':sha256(path),'bytes':path.stat().st_size,'known_pixels':metric['known_pixels']})
    record={'at':now(),'registry_id':registry['registry_id'],'result_sha256':sha256(destination/'result.json'),
        'files':proof,'status':'retirement_prepared','scope':'Derived per-image AP intermediates only; frozen model/source images, predictions, complete native summaries and fixed figure maps retained. Exact arrays can be re-generated from the immutable model/cohort.'}
    save_json(marker,record)
    for path in files:path.unlink()
    record['status']='retired';save_json(marker,record);return record


def execute():
    registry=read_json(FINAL/'frozen_registry.json')
    if gate.registry_digest(registry)!=registry['registry_id']:raise RuntimeError('Frozen registry changed')
    jobs=registry['analysis_plan']['confirmation_jobs']
    if len({(j['model_id'],j['cohort']) for j in jobs})!=len(jobs):raise RuntimeError('Repeated frozen evaluation jobs')
    lock=(FINAL/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    handles=[]
    # Do not compete with any own source/benchmark worker. Status alone is not a lock.
    for study in list(STUDIES.values())+['inference_efficiency_v1']:
        handle=(BASE/study/'runner.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX);handles.append(handle)
    gpu=(BASE/'gpu_work_queue.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX)
    completed=0
    try:
        for job in jobs:
            if shutil.disk_usage(ROOT).free<registry['analysis_plan']['storage_policy']['minimum_free_GiB']*1024**3:
                raise RuntimeError('Insufficient storage; preserve observations and resume exact frozen registry after resolving space')
            save_json(FINAL/'execution_status.json',{'at':now(),'state':'running','pid':os.getpid(),
                'completed_jobs':completed,'planned_jobs':len(jobs),'current':job})
            evaluate(FINAL,job['model_id'],job['cohort'],registry['analysis_plan']['inference_device'])
            verify_prediction_result(FINAL,registry,job)
            if registry['analysis_plan']['storage_policy']['retire_verified_native_pixel_cache']:
                retire_native_cache(FINAL,registry,job)
            completed+=1;gc.collect()
        save_json(FINAL/'execution_status.json',{'at':now(),'state':'complete','completed_jobs':completed,'planned_jobs':len(jobs),
            'scope':'Frozen predictions completed; scientific analysis/manuscript readiness still requires audit and interpretation'})
    except BaseException as error:
        save_json(FINAL/'execution_status.json',{'at':now(),'state':'failed','completed_jobs':completed,'planned_jobs':len(jobs),
            'error':str(error),'traceback':traceback.format_exc(),'observed_data_policy':'Partial failures count as observed; do not replace model/cohort/threshold/evaluator choices'})
        raise


def audit():
    registry=read_json(FINAL/'frozen_registry.json')
    if gate.registry_digest(registry)!=registry['registry_id']:raise RuntimeError('Registry digest differs')
    records=[]
    for job in registry['analysis_plan']['confirmation_jobs']:
        path=FINAL/'results'/job['model_id']/job['cohort']/'result.json'
        if not path.exists():raise RuntimeError('Confirmation is incomplete; audit never opens a new job')
        _,result,_=verify_prediction_result(FINAL,registry,job)
        records.append({'model_id':job['model_id'],'cohort':job['cohort'],'analysis_stratum':job['analysis_stratum'],
            'result':gate.bind_file(path),'predictions_sha256':result['predictions_sha256']})
    proof={'at':now(),'registry_id':registry['registry_id'],'verified_jobs':len(records),'jobs':records,
        'scope':'Complete ordered IDs/labels, saved scores, strict source decisions and metric identity checked. This does not replace localization/uncertainty/failure analysis or establish publication readiness.'}
    save_json(FINAL/'execution_audit.json',proof);return proof
