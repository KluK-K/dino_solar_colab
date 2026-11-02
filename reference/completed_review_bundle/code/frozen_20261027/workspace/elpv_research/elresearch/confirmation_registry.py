"""Immutable confirmation bindings. Building code does not authorize opening test data."""
from pathlib import Path
import fcntl
import math
import re
from importlib.metadata import version

from .common import ROOT, digest, now, read_json, save_json, sha256


ALLOWED_COHORTS = {
    'scdd_publication_v1': ('test', 117),
    'ucf_reported_real_confirmation_v1': ('external', 507),
    'pvelad_restricted_confirmation_v2': ('external', 12326),
}
REQUIRED_DEVELOPMENT = {'image', 'classical', 'dense', 'ucf_reference_transfer', 'cnn_stability', 'representation_scale','mechanistic','published_methods'}
EVALUATOR_MODULES = ['confirmation_registry.py','confirmation_data.py','confirmation_predict.py',
    'confirmation_evaluate.py','streaming_inference.py','classical_features.py','publication_models.py',
    'publication_dense_train.py','publication_dense_models.py','publication_dense_data.py','publication_normal_bank.py',
    'publication_train.py','full_models.py','adapt.py','image_classification.py','classification_heads.py',
    'encoders.py','data.py','metrics.py','localization_metrics.py','pvelad_data.py','common.py']
EVALUATOR_MODULES += ['representation_features.py','representation_training.py']
EVALUATOR_MODULES += ['mechanistic_models.py','mechanistic_train.py','mechanistic_evaluate.py',
    'postadapt_encoder.py','normal_distance_fit.py','packed_normal_bank.py']
EVALUATOR_MODULES += ['published_semisup.py','published_psmt_architecture.py','published_psmt_resnet.py',
    'published_weights.py','published_archive.py','lossless_state.py']
EVALUATOR_MODULES += ['inference_efficiency.py','efficiency_evidence.py']
RUNTIME_PACKAGES = ['torch','torchvision','transformers','huggingface-hub','numpy','scipy','scikit-learn','scikit-image','Pillow']


def runtime_versions():
    return {name:version(name) for name in RUNTIME_PACKAGES}


def _canonical_folder(folder):
    expected = ROOT / 'artifacts/publication_20261017/final_confirmation_v1'
    if Path(folder).resolve() != expected.resolve():
        raise PermissionError('Use the single canonical confirmation registry; a new directory cannot reset observation history')


def _path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise PermissionError('Confirmation artifacts must stay inside the research workspace')
    return path


def check_file(binding):
    path = _path(binding['path'])
    if sha256(path) != binding['sha256']:
        raise RuntimeError('Frozen file changed: ' + binding['path'])
    return path


def bind_file(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()): raise PermissionError('External artifact path')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def selection_exposure(protocol_path):
    """Track labels used for hyperparameter selection as well as gradient fitting."""
    protocol=read_json(protocol_path);identities=[];lineage=[]
    for key in ['source_training_ids','source_validation_ids']:
        identities.extend(protocol.get(key,[]))
    calibration=protocol.get('calibration',{})
    if isinstance(calibration,dict):
        for key in ['training_ids','validation_ids']:identities.extend(calibration.get(key,[]))
    if 'calibration_protocol_sha256' in protocol:
        path=ROOT/'artifacts/publication_20261017/calibration_v1/protocol.json'
        if sha256(path)!=protocol['calibration_protocol_sha256']:
            raise RuntimeError('Upstream source hyperparameter calibration changed')
        upstream=read_json(path)
        identities.extend(upstream['source_training_ids']+upstream['source_validation_ids'])
        lineage.append(bind_file(path))
    if 'upstream_dense_protocol' in protocol:
        check_file(protocol['upstream_dense_protocol']);lineage.append(protocol['upstream_dense_protocol'])
    if 'source_hyperparameter_selection' in protocol:
        lineage.append(bind_file(ROOT/protocol['source_hyperparameter_selection']))
    return sorted(set(identities)),lineage


def verify_source_implementation(protocol,run):
    hashes=run.get('scientific_hashes',protocol.get('scientific_hashes',{}))
    if not hashes:raise RuntimeError('Source scientific implementation identity is missing')
    paths=[]
    for name,expected in hashes.items():
        if Path(name).name!=name or not name.endswith('.py'):raise ValueError('Invalid source implementation filename')
        path=ROOT/'elresearch'/name
        if sha256(path)!=expected:raise RuntimeError('Source model implementation changed: '+name)
        paths.append(path)
    return paths


def cohort_binding(name):
    if name not in ALLOWED_COHORTS:
        raise PermissionError('Cohort is not an approved research confirmation view')
    path = ROOT / 'data' / name / 'manifest.json'
    manifest = read_json(path); partition, expected = ALLOWED_COHORTS[name]
    rows = [r for r in manifest['records'] if r['partition'] == partition]
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise RuntimeError('Confirmation cohort identity/count changed')
    if any(r['target'] not in [0, 1] or r.get('dataset') == 'elpv' for r in rows):
        raise PermissionError('Unknown labels or course ELPV data cannot enter confirmation')
    if name=='pvelad_restricted_confirmation_v2' and any(
        r.get('role')!=('known_annotated_crack_positive' if r['target']==1 else 'author_normal_negative') for r in rows):
        raise PermissionError('Restricted PVEL negatives must be author-normal, never missing-XML/other-defect unknowns')
    return {'name': name, 'manifest': bind_file(path), 'partition': partition,
            'images': len(rows), 'positive': sum(r['target'] for r in rows),
            'ordered_ids_sha256': digest([r['id'] for r in rows]),
            'task_scope': ('Known crack/star-crack sensitivity and author-normal FPR; other-defect negatives unknown'
                           if name.startswith('pvelad') else 'Annotated crack presence anywhere in the image field of view'),
            'uncertainty_unit': None if name.startswith('pvelad') else 'provided acquisition family, physical module unverified'}


def model_source_ids(model):
    """Image-only SSL exposure counts for identity/domain leakage, not mask cost."""
    return (model['source_train_ids'] + model['source_validation_ids']
            + model.get('source_unlabelled_ids',[]) + model['hyperparameter_selection_exposure_ids'])


def bind_model(kind, family, seed, run_folder, source_protocol, source_view='primary'):
    if kind not in {'image', 'classical', 'dense'}: raise ValueError('Unknown model kind')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', family): raise ValueError('Invalid family identifier')
    run_folder = Path(run_folder); path = run_folder / 'run.json'; run = read_json(path)
    protocol=read_json(source_protocol)
    if run.get('implementation_smoke') or protocol.get('implementation_smoke'):
        raise PermissionError('Implementation smoke checks are not scientific confirmation models')
    verify_source_implementation(protocol,run)
    architecture=run.get('model',{}).get('architecture')
    if source_view!='primary' and (architecture!='published_psmt_deeplab' or source_view!='ensemble'):
        raise ValueError('Only a predeclared published teacher-ensemble view may replace primary inference')
    source=run
    if architecture=='published_psmt_deeplab':
        if kind!='dense' or source_view not in run['inference_views']:
            raise ValueError('Published architecture requires a verified dense inference view')
        source=run['inference_views'][source_view]
    checkpoint = run_folder / ('model.joblib' if kind == 'classical' else 'best.pt')
    if sha256(checkpoint) != run['checkpoint_sha256']:
        raise RuntimeError('Selected model checkpoint changed')
    if seed != run['seed']: raise ValueError('Selected model seed differs')
    threshold_key = {'image': 'source_calibrated_logit_threshold',
                     'classical': 'source_calibrated_margin_threshold', 'dense': 'source_logit_threshold'}[kind]
    threshold, temperature = source[threshold_key], source['source_temperature']
    if not math.isfinite(threshold) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Nonfinite source operating point')
    if kind == 'image':
        verification = bind_file(run_folder / 'source_reload_verification.json')
        check = read_json(check_file(verification))
        if (check['status'] != 'verified' or check['run_sha256'] != sha256(path)
                or check['checkpoint_sha256'] != run['checkpoint_sha256'] or not check['auroc_exactly_preserved']):
            raise RuntimeError('Image model lacks valid source-only reload verification')
    elif kind == 'classical':
        if not run['source_reload_verified'] or run['source_reload_maximum_absolute_error'] != 0:
            raise RuntimeError('Classical saved-model predictions were not verified')
        verification = None
    else:
        check = source['checkpoint_reload']
        if (not check['fresh_model'] or check['logit_max_abs_error'] > 2e-5
                or check['sampled_pixel_probability_max_abs_error'] > 2e-5):
            raise RuntimeError('Dense checkpoint fails fresh-model verification')
        verification = None
    exposure,lineage=selection_exposure(source_protocol)
    dependencies=[]
    if architecture=='published_psmt_deeplab':
        spec=run['model'];basis=spec['archive_basis'];backbone=run['encoder']['backbone']
        if basis['backbone_sha256']!=spec['backbone_sha256'] or backbone['sha256']!=spec['backbone_sha256']:
            raise RuntimeError('Published archive initialization identities disagree')
        for binding in [basis,backbone]:
            dependency={k:binding[k] for k in ['path','sha256']}
            check_file(dependency);dependencies.append(dependency)
    unlabelled=run.get('unlabelled_source_ids',[])
    if set(unlabelled)&set(run['training_ids']+run['validation_ids']):
        raise PermissionError('Unlabelled source image identities overlap labelled training/validation')
    if architecture=='dinov3_postadapt':
        spec=run['model'];folder=_path(spec['adapted_encoder_path'])
        for name,key in [('model.safetensors','adapted_weights_sha256'),('config.json','adapted_config_sha256')]:
            if sha256(folder/name)!=spec[key]:raise RuntimeError('Frozen adapted encoder changed')
            dependencies.append(bind_file(folder/name))
        certificate=read_json(folder/'encoder_export.json')
        if (certificate['weights_sha256']!=spec['adapted_weights_sha256']
                or certificate['config_sha256']!=spec['adapted_config_sha256']
                or certificate['signature']['parent_run_sha256']!=spec['adaptation_parent_run_sha256']
                or certificate['merge_max_abs_error']>3e-5 or certificate['fresh_encoder_reload_max_abs_error']>2e-5):
            raise RuntimeError('Adapted-encoder merge/reload proof does not match source')
        dependencies.append(bind_file(folder/'encoder_export.json'))
        parent_path=_path(spec['adaptation_parent_run'])
        if sha256(parent_path)!=spec['adaptation_parent_run_sha256']:raise RuntimeError('Adaptation parent changed')
        parent=read_json(parent_path)
        if parent['training_ids']!=run['training_ids'] or parent['validation_ids']!=run['validation_ids']:
            raise PermissionError('Post-adaptation head uses different source annotation identities')
        if sha256(parent_path.parent/'best.pt')!=parent['checkpoint_sha256']:raise RuntimeError('Adaptation parent checkpoint changed')
        dependencies.extend([bind_file(parent_path),bind_file(parent_path.parent/'best.pt')])
    recipe_path=run_folder/'normal_bank_recipe.json'
    if recipe_path.exists():
        recipe=read_json(recipe_path)
        if recipe.get('format')=='canonical_orientation_memory_v2':
            dependencies.append(recipe['payload']);check_file(recipe['payload'])
        else:
            for row in recipe['rows']:
                dependency={'path':str(Path(recipe['cache_directory'])/(row['cache_key']+'.pt')),'sha256':row['payload_sha256']}
                check_file(dependency);dependencies.append(dependency)
    return {'id': f'{family}_seed{seed}', 'family': family, 'seed': seed, 'kind': kind,
            'architecture':architecture,'source_view':source_view,'frozen_model_dependencies':dependencies,
            'run': bind_file(path), 'source_protocol': bind_file(source_protocol),
            'checkpoint': bind_file(checkpoint), 'reload_verification': verification,
            'source_threshold': float(threshold), 'source_temperature': float(temperature),
            'source_train_ids': run['training_ids'], 'source_validation_ids': run['validation_ids'],
            'source_unlabelled_ids':unlabelled,
            'hyperparameter_selection_exposure_ids':exposure,'selection_lineage':lineage,
            'training_fraction': run['training_budget_fraction'],
            'normal_bank_recipe': bind_file(run_folder / 'normal_bank_recipe.json') if (run_folder / 'normal_bank_recipe.json').exists() else None}


def validate_development_evidence(evidence):
    """Counts/paths are checked, not treated as completed just because 'ready' is set."""
    if set(evidence) != REQUIRED_DEVELOPMENT:
        raise RuntimeError('Image, classical, dense, UCF-transfer, CNN-stability, representation/scale, local/post-adaptation mechanism and matched published-method development must all be complete')
    for name, binding in evidence.items():
        status = read_json(check_file(binding['status']))
        protocol_path = check_file(binding['protocol']); protocol = read_json(protocol_path)
        planned = (protocol['planned_real_domain_runs'] + protocol['calibration']['planned_runs'] if name == 'dense'
                   else protocol['development_logical_runs'] if name == 'classical' else protocol['planned_runs'])
        if (planned <= 0 or binding['expected_runs'] != planned or status['state'] != 'complete'
                or status.get('completed_runs') != planned):
            raise RuntimeError('Development study is incomplete: ' + name)
        if name == 'classical' and status.get('source_calibration_complete') != protocol['source_calibration_trials']:
            raise RuntimeError('Classical source calibration is incomplete')
        runs = list(protocol_path.parent.glob(binding['completed_run_glob']))
        if len(runs) != binding['expected_runs'] or any(not p.is_file() for p in runs):
            raise RuntimeError('Completed run files do not cover registered development: ' + name)
        # Every run must still bind to an actual unchanged fitted model.
        for path in runs:
            run = read_json(path)
            checkpoint = path.parent / ('model.joblib' if name in {'classical','representation_scale'} else 'best.pt')
            if sha256(checkpoint) != run['checkpoint_sha256']:
                raise RuntimeError('Development checkpoint missing or changed: ' + str(path))


def registry_digest(registry):
    return digest({k: v for k, v in registry.items() if k != 'registry_id'})


def verified_inference_efficiency():
    """Finish the predeclared compute comparison before opening blind results."""
    folder=ROOT/'artifacts/publication_20261017/inference_efficiency_v1'
    if not all((folder/name).exists() for name in ['protocol.json','status.json','model_bindings.json','verification.json']):
        raise RuntimeError('Complete source-only inference efficiency must be measured and verified before confirmation')
    protocol=read_json(folder/'protocol.json')
    if protocol['planned_jobs']!=97 or {k:len(v) for k,v in protocol['cohorts'].items()}!={'scdd_publication_v1':351,'ucf':430}:
        raise RuntimeError('Full97-system/source-cohort inference-efficiency scope differs')
    for name,expected in protocol['scientific_hashes'].items():
        if Path(name).name!=name or sha256(ROOT/'elresearch'/name)!=expected:
            raise RuntimeError('Registered inference-efficiency implementation changed')
    if sha256(ROOT/'run_inference_efficiency.py')!=protocol['runner_sha256']:
        raise RuntimeError('Registered efficiency runner changed')
    from .efficiency_evidence import verify_completed_benchmark
    proof=verify_completed_benchmark(folder)
    if proof!=read_json(folder/'verification.json'):raise RuntimeError('Full efficiency verification changed')
    return {'verified_jobs':proof['verified_jobs'],
        **{key:bind_file(folder/name) for key,name in [('protocol','protocol.json'),('status','status.json'),
            ('model_bindings','model_bindings.json'),('verification','verification.json')]}}


def freeze_registry(folder, models, cohorts, development_evidence, analysis_plan, evaluator_files, source_identity_index):
    """Only call after development decisions. No real registry is created during tests/drafting."""
    _canonical_folder(folder)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / 'frozen_registry.json').exists() or (folder / 'observation_opened.json').exists():
            raise RuntimeError('Confirmation is already frozen/opened; cannot replace its choices')
        validate_development_evidence(development_evidence)
        efficiency=verified_inference_efficiency()
        if not models or len({m['id'] for m in models}) != len(models): raise ValueError('Unique selected models required')
        if not cohorts or len({c['name'] for c in cohorts}) != len(cohorts): raise ValueError('Unique approved cohorts required')
        if not {'primary_comparisons', 'primary_endpoints', 'selection_rationale', 'uncertainty', 'limits'} <= set(analysis_plan):
            raise ValueError('Freeze comparisons, endpoints, source-only rationale, uncertainty and limits')
        if not all(analysis_plan[k] for k in ['primary_comparisons','primary_endpoints','selection_rationale','uncertainty']):
            raise ValueError('Confirmation decisions must be concrete, not empty placeholders')
        if analysis_plan.get('inference_device') not in {'cpu','mps','cuda'}:
            raise ValueError('Freeze the actual inference device, not auto')
        source = read_json(check_file(source_identity_index))
        for binding in source['source_manifests']: check_file(binding)
        if digest({k:v for k,v in source.items() if k!='signature'}) != source['signature']:
            raise RuntimeError('Development source identity index changed')
        source_rows = {r['id']:r for r in source['records']}
        for family in {m['family'] for m in models}:
            group = [m for m in models if m['family'] == family]
            seeds = sorted(m['seed'] for m in group)
            deterministic_full = all((m['kind']=='classical' or m.get('architecture')=='dinov3_normal_distance')
                                     and m['training_fraction']==1 for m in group)
            if deterministic_full and seeds != [11]:
                raise ValueError('A deterministic full-budget probe is counted once, with seed11')
            if not deterministic_full and seeds != [11, 22, 33]:
                raise ValueError('Each stochastic/subset family requires seeds 11/22/33')
        for model in models:
            for key in ['run', 'source_protocol', 'checkpoint', 'reload_verification', 'normal_bank_recipe']:
                if model.get(key): check_file(model[key])
            for binding in model['selection_lineage']:check_file(binding)
            for binding in model.get('frozen_model_dependencies',[]):check_file(binding)
            canonical = bind_model(model['kind'],model['family'],model['seed'],
                                   check_file(model['run']).parent,check_file(model['source_protocol']),
                                   model.get('source_view','primary'))
            if canonical != model: raise RuntimeError('Selected model/threshold differs from verified source-run binding')
            known = model_source_ids(model)
            if not set(known) <= set(source_rows):
                raise PermissionError('Selected model source identities are outside audited development')
            if set(model['source_train_ids']) & set(model['source_validation_ids']):
                raise PermissionError('Source training/validation identities overlap')
        for cohort in cohorts:
            if cohort != cohort_binding(cohort['name']): raise RuntimeError('Cohort binding changed')
            rows = read_json(check_file(cohort['manifest']))['records']
            target_ids = {r['id'] for r in rows if r['partition'] == cohort['partition']}
            if any(target_ids & set(model_source_ids(m)) for m in models):
                raise PermissionError('Selected model fitted or selected hyperparameters on confirmation identities')
            targets = [r for r in rows if r['partition'] == cohort['partition']]
            target_orbits = {r['d4_pixel_sha256'] for r in targets}
            target_groups = {(r['dataset'],r['split_group']) for r in targets if r.get('split_group') is not None}
            for model in models:
                source_ids = model_source_ids(model)
                selected_source = [source_rows[i] for i in source_ids]
                if target_orbits & {r['d4_pixel_sha256'] for r in selected_source}:
                    raise PermissionError('Exact D4 source/confirmation duplicate leakage')
                if target_groups & {(r['dataset'],r['split_group']) for r in selected_source}:
                    raise PermissionError('Source/confirmation acquisition-family overlap')
        source_files=set()
        for model in models:
            source_files.update(verify_source_implementation(read_json(check_file(model['source_protocol'])),
                                                              read_json(check_file(model['run']))))
        evaluator_files = [bind_file(path) for path in sorted(set(Path(p).resolve() for p in evaluator_files)
                            | {ROOT/'elresearch'/name for name in EVALUATOR_MODULES} | source_files)]
        if not evaluator_files: raise ValueError('The evaluator implementation must also be frozen')
        registry = {'version': 1, 'frozen_at': now(), 'models': models, 'cohorts': cohorts,
                    'development_evidence': development_evidence, 'analysis_plan': analysis_plan,
                    'inference_efficiency_evidence':efficiency,
                    'source_identity_index': source_identity_index,
                    'runtime_versions':runtime_versions(),
                    'evaluator_files': evaluator_files,
                    'observation_policy': 'One frozen registry; exact interrupted-job continuation only. No replacement model, threshold, cohort or evaluation code after opening.'}
        registry['registry_id'] = registry_digest(registry)
        save_json(folder / 'frozen_registry.json', registry)
        return registry


def authorize_rows(folder, model_id, cohort_name):
    """Verify every frozen binding before any image read/model call; record opening."""
    _canonical_folder(folder)
    folder = Path(folder)
    with (folder / 'registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = read_json(folder / 'frozen_registry.json')
        if registry_digest(registry) != registry['registry_id']: raise RuntimeError('Frozen confirmation registry modified')
        if runtime_versions()!=registry['runtime_versions']:raise RuntimeError('Frozen inference dependency versions changed')
        source = read_json(check_file(registry['source_identity_index']))
        for binding in source['source_manifests']: check_file(binding)
        for binding in registry['evaluator_files']: check_file(binding)
        for key in ['protocol','status','model_bindings','verification']:
            binding=registry['inference_efficiency_evidence'].get(key)
            if binding:check_file(binding)
        model = next((m for m in registry['models'] if m['id'] == model_id), None)
        cohort = next((c for c in registry['cohorts'] if c['name'] == cohort_name), None)
        if model is None or cohort is None: raise PermissionError('Unregistered model/cohort combination')
        for key in ['run', 'source_protocol', 'checkpoint', 'reload_verification', 'normal_bank_recipe']:
            if model.get(key): check_file(model[key])
        for binding in model['selection_lineage']:check_file(binding)
        for binding in model.get('frozen_model_dependencies',[]):check_file(binding)
        if cohort != cohort_binding(cohort_name): raise RuntimeError('Frozen cohort changed')
        manifest = read_json(check_file(cohort['manifest']))
        rows = [r for r in manifest['records'] if r['partition'] == cohort['partition']]
        marker = folder / 'observation_opened.json'
        if marker.exists():
            if read_json(marker)['registry_id'] != registry['registry_id']:
                raise RuntimeError('Observation already opened under a different registry')
        else:
            save_json(marker, {'opened_at': now(), 'registry_id': registry['registry_id'],
                               'first_model': model_id, 'first_cohort': cohort_name,
                               'meaning': 'Confirmation access begins here, before any model scores; partial failures count as observed.'})
        return registry, model, cohort, rows
