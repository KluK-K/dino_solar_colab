"""Analyze every frozen primary contrast after all registered predictions exist.

Reads saved outputs only: no encoder, threshold fitting, target-model selection,
pixel opening, or imaginary PVEL acquisition groups. Intervals condition on the
fitted seed pairs; deterministic controls remain one fitted predictor.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import ROOT, read_json, save_json, sha256, digest, now
from . import confirmation_registry as gate
from . import confirmation_workflow as workflow
from .metrics import classification
from .publication_train import probability_metrics


IMAGE_ENDPOINTS = {'AUROC', 'average_precision', 'recall_at_source_threshold',
    'review_rate_at_source_threshold', 'false_positive_rate_at_source_threshold',
    'known_crack_recall', 'author_normal_FPR'}
LOCAL_ENDPOINTS = {'positive_image_Dice', 'positive_image_IoU',
    'normal_structure_pixel_FPR', 'box_point_hit_rate', 'pooled_native_pixel_AP'}


class EndpointUnavailable(ValueError):
    """An annotation/class denominator is absent; retain this declared contrast."""


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def endpoint_value(record, endpoint, weight=None):
    """Prespecified endpoint in its original unit, using image/family weights."""
    y = np.asarray(record['y']); scores = np.asarray(record['logits'])
    weight = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    if (weight.shape != y.shape or (weight < 0).any() or not np.isfinite(weight).all()
        or not weight.sum()):
        raise ValueError('Invalid image/family weights')
    if endpoint in IMAGE_ENDPOINTS:
        positive = float(weight[y == 1].sum()); negative = float(weight[y == 0].sum())
        predicted = scores > record['source_threshold']
        if endpoint == 'AUROC':
            return float(roc_auc_score(y, scores, sample_weight=weight)) if positive and negative else None
        if endpoint == 'average_precision':
            return float(average_precision_score(y, scores, sample_weight=weight)) if positive else None
        if endpoint in {'recall_at_source_threshold', 'known_crack_recall'}:
            return _ratio(float(weight[(y == 1) & predicted].sum()), positive)
        if endpoint in {'false_positive_rate_at_source_threshold', 'author_normal_FPR'}:
            return _ratio(float(weight[(y == 0) & predicted].sum()), negative)
        return float(weight[predicted].sum() / weight.sum())
    if endpoint not in LOCAL_ENDPOINTS:
        raise ValueError('Unknown prespecified endpoint')
    local = record.get('localization') or {}
    if endpoint == 'pooled_native_pixel_AP':
        if record.get('analysis_stratum', 'whole') != 'whole' or not np.all(weight == 1) or record.get('has_population_exclusions'):
            raise ValueError('Pooled native AP supports a whole-cohort point only; never substitute per-image AP')
        return local.get('pooled_native_pixel_AP')
    by_id = {value['id']: value for value in local.get(
        'box_per_image' if endpoint == 'box_point_hit_rate' else 'native_pixel_per_image', [])}
    numerator = denominator = 0.
    for identifier, w in zip(record['ids'], weight):
        value = by_id.get(str(identifier))
        if value is None or not w:
            continue
        if endpoint == 'box_point_hit_rate':
            numerator += w * value['maximum_point_in_target_box']; denominator += w
        elif endpoint == 'normal_structure_pixel_FPR':
            count = value.get('negative_image_normal_structure_pixels') or 0
            if count:
                numerator += w * value['negative_image_normal_structure_false_positive_pixels']
                denominator += w * count
        elif value['crack_positive']:
            numerator += w * value['dice' if endpoint == 'positive_image_Dice' else 'iou']
            denominator += w
    return _ratio(numerator, denominator)


def _deterministic(record):
    model = record['model']
    return model['training_fraction'] == 1. and (
        model['kind'] == 'classical' or model.get('architecture') == 'dinov3_normal_distance')


def pair_records(first, second):
    """Match stochastic seeds, or reuse one explicitly deterministic predictor."""
    if not first or not second:
        raise ValueError('Missing fitted predictors')
    a = {r['model']['seed']: r for r in first}; b = {r['model']['seed']: r for r in second}
    if len(a) != len(first) or len(b) != len(second):
        raise ValueError('Duplicate fitted training seed')
    if set(a) == set(b):
        pairs = [(a[s], b[s]) for s in sorted(a)]
    elif len(a) == 1 and _deterministic(first[0]):
        pairs = [(first[0], b[s]) for s in sorted(b)]
    elif len(b) == 1 and _deterministic(second[0]):
        pairs = [(a[s], second[0]) for s in sorted(a)]
    else:
        raise ValueError('Unpaired stochastic fits cannot be made into matched replications')
    reference = pairs[0][0]
    for left, right in pairs:
        for row in [left, right]:
            for key in ['ids', 'y', 'groups']:
                if not np.array_equal(row[key], reference[key]):
                    raise ValueError('Primary contrast is not aligned by image, label and group')
            if not np.isfinite(row['logits']).all():
                raise ValueError('Nonfinite frozen predictions')
    return pairs


def paired_endpoint(first, second, endpoint, repetitions=1000, seed=20261017,
                    acquisition_groups_available=True):
    pairs = pair_records(first, second)
    values = [(endpoint_value(a, endpoint), endpoint_value(b, endpoint)) for a, b in pairs]
    if any(x is None for pair in values for x in pair):
        raise EndpointUnavailable('Prespecified endpoint lacks annotation/class support; report this failure, never choose a substitute')
    difference = np.array([b-a for a, b in values])
    result = {'difference': 'second minus first; mean of fitted-pair differences, not an ensemble',
        'first_mean': float(np.mean([v[0] for v in values])),
        'second_mean': float(np.mean([v[1] for v in values])),
        'mean_pair_difference': float(difference.mean()),
        'fitted_pairs': [{'first_model_id': a['model']['id'], 'second_model_id': b['model']['id'],
            'first_seed': a['model']['seed'], 'second_seed': b['model']['seed'],
            'first_value': x, 'second_value': z, 'difference': z-x}
            for (a, b), (x, z) in zip(pairs, values)],
        'first_unique_fitted_predictors': len(first), 'second_unique_fitted_predictors': len(second),
        'pair_difference_sample_sd': float(difference.std(ddof=1)) if len(pairs) > 1 else None,
        'acquisition_family_percentile_ci95': None,
        'limits': ['Conditional on fitted models. Reusing a deterministic predictor does not create new training replications.',
            'No multiplicity-adjusted hypothesis test or population-over-training-runs interval is asserted.']}
    if endpoint == 'pooled_native_pixel_AP':
        result['interval_status'] = 'Point only: pooled native AP is not an average of per-image AP; individual pixels are never bootstrapped.'
        return result
    groups = np.asarray(pairs[0][0]['groups'])
    if not acquisition_groups_available:
        result['interval_status'] = 'Unavailable: physical acquisition groups are not supplied; filenames are not invented cluster identities.'
        return result
    if any(g is None or not str(g).strip() or str(g).lower() in {'none', 'unknown', 'nan'} for g in groups):
        raise ValueError('Missing acquisition groups cannot produce a family interval')
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        result['interval_status'] = 'Unavailable: fewer than two provided acquisition families.'
        return result
    if not isinstance(repetitions, int) or repetitions < 20:
        raise ValueError('Insufficient registered bootstrap repetitions')
    rng = np.random.default_rng(seed); replicates = []
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        weights = counts[inverse]
        values = [(endpoint_value(a, endpoint, weights), endpoint_value(b, endpoint, weights)) for a, b in pairs]
        if any(x is None for pair in values for x in pair):
            continue
        replicates.append(np.mean([b-a for a, b in values]))
    if len(replicates) < max(20, repetitions // 2):
        result['interval_status'] = 'Unavailable: insufficient annotated/two-class acquisition-family replicates.'
    else:
        result['acquisition_family_percentile_ci95'] = np.quantile(replicates, [.025, .975]).tolist()
        result['interval_status'] = 'Conditional paired acquisition-family percentile interval'
    result.update(acquisition_families=len(unique), requested_repetitions=repetitions,
        valid_repetitions=len(replicates), bootstrap_seed=seed)
    return result


def load_record(folder, registry, job, stratum=None,subset='primary'):
    destination, result, rows = workflow.verify_prediction_result(folder, registry, job)
    model = next(m for m in registry['models'] if m['id'] == job['model_id'])
    stratum = job['analysis_stratum'] if stratum is None else stratum
    from .confirmation_population import checked_policy,selected_rows
    binding=registry['analysis_plan'].get('cohort_analysis_policy_binding')
    population=checked_policy(binding) if binding else None
    eligible,excluded=selected_rows(rows,job['cohort'],population,subset)
    eligible_ids={r['id'] for r in eligible}
    keep = np.array([row['id'] in eligible_ids and (stratum == 'whole' or row.get('acquisition_source') == stratum) for row in rows])
    if not keep.any():
        raise ValueError('Frozen acquisition-source stratum has no images')
    with np.load(destination/'predictions.npz', allow_pickle=False) as value:
        record = {key: value[key][keep].copy() for key in ['ids', 'y', 'logits']}
    record.update(groups=np.array([row.get('split_group') for row, selected in zip(rows, keep) if selected], dtype=object),
        model=model, source_threshold=model['source_threshold'], localization=result.get('localization'),
        analysis_stratum=stratum, result=result, result_binding=gate.bind_file(destination/'result.json'),
        row_metadata=[row for row,selected in zip(rows,keep) if selected],analysis_population=subset,
        has_population_exclusions=bool(excluded),underlying_scored_images=len(rows),population_excluded_ids=excluded)
    return record


def spatial_summary(record):
    return {endpoint:endpoint_value(record,endpoint) for endpoint in sorted(LOCAL_ENDPOINTS)
        if record['localization'] and (endpoint!='pooled_native_pixel_AP' or
            (record['analysis_stratum']=='whole' and not record.get('has_population_exclusions')))}


def restricted_summary(record):
    metrics=classification(record['y'],record['logits'],record['source_threshold'])
    return {'known_crack_recall':metrics['recall'],'author_normal_FPR':metrics['fp']/(metrics['fp']+metrics['tn']),
        'known_crack_images':metrics['positive'],'author_normal_images':metrics['n']-metrics['positive']}


def analyze(folder=None):
    """Requires a complete verified execution audit; never initiates inference."""
    folder = workflow.FINAL if folder is None else Path(folder)
    registry = read_json(folder/'frozen_registry.json')
    if gate.registry_digest(registry) != registry['registry_id']:
        raise RuntimeError('Frozen registry changed')
    for binding in registry['evaluator_files']:
        gate.check_file(binding)
    execution = read_json(folder/'execution_audit.json')
    if execution['registry_id'] != registry['registry_id']:
        raise RuntimeError('Execution audit belongs to another registry')
    jobs = registry['analysis_plan']['confirmation_jobs']
    expected = {(j['model_id'], j['cohort'], j['analysis_stratum']) for j in jobs}
    audited = {(j['model_id'], j['cohort'], j['analysis_stratum']) for j in execution['jobs']}
    if expected != audited or execution['verified_jobs'] != len(jobs) or len(execution['jobs']) != len(jobs):
        raise RuntimeError('Every frozen job must be audited before primary analysis')
    for item in execution['jobs']:
        gate.check_file(item['result'])
    plan = read_json(gate.check_file(registry['analysis_plan']['statistics_plan_binding']))
    from .secondary_failure_analysis import checked_failure_policy,native_failure_summary
    failure_binding=registry['analysis_plan'].get('secondary_native_failure_policy_binding')
    failure_policy=checked_failure_policy(failure_binding) if failure_binding else None
    per_job = []
    for job in jobs:
        r = load_record(folder, registry, job)
        spatial=spatial_summary(r)
        sensitivities={}
        population_binding=registry['analysis_plan'].get('cohort_analysis_policy_binding')
        if population_binding:
            from .confirmation_population import checked_policy
            population=checked_policy(population_binding)
            if job['cohort'] in population['cohorts']:
                for subset in population['cohorts'][job['cohort']]['subsets']:
                    if subset=='primary':continue
                    alternative=load_record(folder,registry,job,subset=subset)
                    sensitivities[subset]={'metrics':classification(alternative['y'],alternative['logits'],alternative['source_threshold']),
                        'localization_point_estimates':spatial_summary(alternative),
                        'scope':'Same scored pool and fitted predictor; predeclared secondary sensitivity, not an independent replication.'}
        line_sensitivity=None
        if job['cohort'].startswith('pvelad'):
            line=np.array([row.get('target_line_crack') is not None for row in r['row_metadata']])
            line_sensitivity={'excluded_star_only_images':int((~line).sum()),
                'metrics':classification(r['y'][line],r['logits'][line],r['source_threshold']),
                'scope':'Line-only sensitivity within the frozen primary population; normal negatives retained.'}
        per_job.append({'model_id': job['model_id'], 'cohort': job['cohort'], 'analysis_stratum': job['analysis_stratum'],
            'analysis_role': job['analysis_role'], 'result': r['result_binding'],
            'metrics': classification(r['y'], r['logits'], r['source_threshold']),
            'source_temperature_probability_metrics': probability_metrics(r['y'], r['logits'], r['model']['source_temperature']),
            'development_exposure': r['result']['development_exposure'],
            'task_scope': r['result']['task_scope'],
            'primary_restricted_endpoints': restricted_summary(r) if job['cohort'].startswith('pvelad') else None,
            'secondary_composition_dependent_metrics': r['result'].get('secondary_composition_dependent_metrics'),
            'line_crack_only_sensitivity':line_sensitivity,
            'primary_population':{'analyzed_images':len(r['ids']),'underlying_scored_images':r['underlying_scored_images'],
                'excluded_ids':r['population_excluded_ids'],'policy':population_binding},
            'population_sensitivities':sensitivities,
            'localization_point_estimates': spatial,
            'secondary_native_failure_analysis':native_failure_summary(r,job['cohort'],failure_policy),
            'localization_limit': 'Pooled AP is whole-cohort only; box-point hit rate is not segmentation AP. Null endpoints lack relevant annotation support.',
            'scope': 'Metrics for this declared model and stratum. No AUROC pooling across separately trained models.'})
    contrasts = []
    for comparison in registry['analysis_plan']['primary_comparisons']:
        groups = []
        for family in [comparison['first_family'], comparison['second_family']]:
            selected = [m for m in registry['models'] if m['family'] == family]
            groups.append([load_record(folder, registry,
                next(j for j in jobs if j['model_id'] == m['id'] and j['cohort'] == comparison['cohort']),
                comparison['analysis_stratum']) for m in selected])
        try:
            estimate = paired_endpoint(*groups, comparison['endpoint'], repetitions=plan['repetitions'], seed=plan['seed'],
                acquisition_groups_available=not comparison['cohort'].startswith('pvelad'))
            contrasts.append({'declaration': comparison, 'status': 'estimated', 'estimate': estimate})
        except EndpointUnavailable as error:
            contrasts.append({'declaration': comparison, 'status': 'not_estimable', 'estimate': None,
                'reason': str(error), 'substitute_endpoint_selected': False})
    inputs = {'registry_id': registry['registry_id'], 'execution_audit': gate.bind_file(folder/'execution_audit.json'),
        'analysis_code_sha256': sha256(__file__), 'statistics_plan': registry['analysis_plan']['statistics_plan_binding']}
    proof = {'at': now(), 'analysis_id': digest(inputs), 'inputs': inputs, 'jobs': per_job,
        'study_wide_design_context':registry['analysis_plan'].get('study_wide_design_context'),
        'frozen_study_limits':registry['analysis_plan'].get('limits',[]),
        'primary_comparisons': contrasts, 'all_frozen_primary_comparisons_reported': True,
        'scope': 'Saved frozen predictions only; no model/threshold fitting, new test job, or acceptance/novelty claim.'}
    destination = folder/'primary_analysis.json'
    if destination.exists():
        old = read_json(destination)
        if {k:v for k,v in old.items() if k!='at'} != {k:v for k,v in proof.items() if k!='at'}:
            raise RuntimeError('Preserve the prior primary analysis; a changed analysis requires an explicit amendment')
        return old
    save_json(destination, proof)
    return proof
