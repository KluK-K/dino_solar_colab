"""Paired saved-output localization analysis for complete development cells.

No training, model inference, raw-image access, threshold choice or final-test
access. Conditional capture-family intervals are exploratory, particularly on
validation images that selected the checkpoints. Pooled AP remains point-only.
"""
import os

import numpy as np

from elresearch.common import ROOT, digest, now, read_json, save_json, sha256
from elresearch.confirmation_analysis import endpoint_value, paired_endpoint, EndpointUnavailable

BASE = ROOT / 'artifacts/publication_20261017'
OUTPUT = BASE / 'dense_development_uncertainty_v1'
CONTRASTS = [
    ('dinov3_dense_frozen', 'dinov3_dense_reference'),
    ('deeplabv3plus_full', 'dinov3_dense_reference'),
    ('dinov3_dense_lora', 'dinov3_dense_reference'),
    ('dinov3_dense_frozen', 'dinov3_dense_lora'),
    ('deeplabv3plus_full', 'dinov3_dense_lora'),
    ('dinov3_dense_frozen', 'dinov3_dense_full'),
    ('deeplabv3plus_full', 'dinov3_dense_full'),
    ('dinov3_reference_no_position', 'dinov3_dense_reference'),
    ('dinov3_reference_mean_evidence', 'dinov3_dense_reference'),
]
ENDPOINTS = ['AUROC', 'positive_image_Dice', 'normal_structure_pixel_FPR',
             'recall_at_source_threshold', 'review_rate_at_source_threshold',
             'pooled_native_pixel_AP']


def saved_record(run, evaluated, arrays, family, expected_ids, expected_groups):
    """Align native summaries with independently saved image IDs/truth/groups."""
    ids = np.asarray(arrays['ids']); y = np.asarray(arrays['y'])
    groups = np.asarray(arrays['groups']); logits = np.asarray(arrays['logits'])
    if len(set(ids.tolist())) != len(ids) or ids.tolist() != list(expected_ids):
        raise ValueError('Saved evaluation identities differ or are duplicated')
    if groups.tolist() != list(expected_groups) or any(x.shape != ids.shape for x in [y, groups, logits]):
        raise ValueError('Saved image/group/score alignment differs')
    if set(np.unique(y)) != {0, 1} or not np.isfinite(logits).all():
        raise ValueError('Finite predictions and both image classes are required')
    local = evaluated['validation_localization'] if 'validation_localization' in evaluated else evaluated['localization']
    summaries = local['per_image']
    if len(summaries) != len(ids) or {r['id'] for r in summaries} != set(ids):
        raise ValueError('Every native image summary must have exactly one saved identity')
    by = {r['id']: r for r in summaries}
    for identifier, target, group in zip(ids, y, groups):
        row = by[str(identifier)]
        if bool(row['crack_positive']) != bool(target) or row['acquisition_family'] != group:
            raise ValueError('Native truth/group differs from image-level saved predictions')
        known, positive, predicted, fp = [row[k] for k in ['known_pixels', 'crack_pixels', 'predicted_crack_pixels', 'false_positive_pixels']]
        tp = predicted - fp
        if (not 0 < known or not 0 <= tp <= positive <= known or not 0 <= fp <= predicted <= known
            or fp > known-positive or bool(positive) != bool(target)):
            raise ValueError('Invalid native foreground/confusion counts')
        expected_dice = 2 * tp / (positive + predicted) if positive + predicted else 1.
        if not np.isclose(row['dice'], expected_dice, atol=1e-12, rtol=0):
            raise ValueError('Saved native Dice disagrees with original confusion counts')
        if row['pixel_threshold'] != .5:
            raise ValueError('Do not retune the native source cutoff')
        structural = row.get('negative_image_normal_structure_pixels')
        structural_fp = row.get('negative_image_normal_structure_false_positive_pixels')
        if structural is not None and (bool(target) or not 0 <= structural <= known-positive
            or structural_fp is None or not 0 <= structural_fp <= min(structural, fp)):
            raise ValueError('Invalid normal-structure subset/confusion counts')
    record = {'ids': ids, 'y': y, 'groups': groups, 'logits': logits,
        'source_threshold': run['source_logit_threshold'], 'analysis_stratum': 'whole',
        'model': {'id': f"{family}_seed{run['seed']}", 'seed': run['seed'],
                  'kind': 'dense', 'training_fraction': run['training_budget_fraction']},
        'localization': {'pooled_native_pixel_AP': local['pooled_pixel_average_precision'],
                         'native_pixel_per_image': summaries}}
    expected = {'positive_image_Dice': local['mean_positive_image_dice'],
                'normal_structure_pixel_FPR': local['pooled_negative_structure_false_positive_fraction']}
    for endpoint, point in expected.items():
        actual = endpoint_value(record, endpoint)
        if (actual is None) != (point is None) or (actual is not None and not np.isclose(actual, point, atol=1e-12, rtol=0)):
            raise ValueError('Native aggregation differs from original full-float evaluation')
    return record


def main():
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Exploratory development analysis must precede final model scoring')
    OUTPUT.mkdir(exist_ok=True)
    statistics = read_json(BASE / 'statistics_plan_v2.json')
    source = BASE / 'dense_development_v1/protocol.json'
    identity_path = BASE / 'source_development_identity_v1.json'
    source_ids = {r['id']: r for r in read_json(identity_path)['records']}
    protocol = read_json(source)
    plan = {'version': 1, 'kind': 'Exploratory complete-cell dense localization comparisons',
        'code_sha256': sha256(__file__), 'endpoint_helper_sha256': sha256(ROOT / 'elresearch/confirmation_analysis.py'),
        'source_protocol_sha256': sha256(source), 'source_identity_sha256': sha256(identity_path),
        'statistics_plan_sha256': sha256(BASE / 'statistics_plan_v2.json'),
        'contrasts': [list(c) for c in CONTRASTS], 'endpoints': ENDPOINTS,
        'repetitions': statistics['repetitions'], 'seed': statistics['seed'],
        'source_outcomes_previously_observed': True,
        'unit': 'Provided capture-family resampling, conditional on all3 paired fitted seeds. Not physical-module identity or a training-run population.',
        'scope': 'Every declared applicable contrast/endpoint in each complete cell; original models/decisions unchanged. Reused source-validation intervals remain exploratory, not confirmation. Whole-cohort pooled pixel AP point only; never bootstrap pixels or replace with image AP.'}
    if (OUTPUT / 'plan.json').exists():
        if read_json(OUTPUT / 'plan.json') != plan:
            raise RuntimeError('Archive and explicitly amend a changed completed analysis')
    else:
        save_json(OUTPUT / 'plan.json', plan)
    aggregate = read_json(BASE / 'dense_development_complete_cells.json')
    outputs = []
    for cell in aggregate['complete_cells']:
        cell_path = ROOT / cell['cell_path']
        if sha256(cell_path) != cell['cell_sha256']:
            raise RuntimeError('Completed all-model cell identity changed')
        identity = {'cell_sha256': cell['cell_sha256'], 'plan_sha256': sha256(OUTPUT / 'plan.json')}
        directory = OUTPUT / digest(identity)[:20]; directory.mkdir(exist_ok=True)
        result_path = directory / 'results.json'
        if result_path.exists():
            saved = read_json(result_path)
            for binding in saved['input_bindings']:
                if sha256(ROOT / binding['path']) != binding['sha256']:
                    raise RuntimeError('Previously analyzed source outputs changed')
            outputs.append({'path': str(result_path.relative_to(ROOT)), 'sha256': sha256(result_path),
                            'fold': cell['fold'], 'budget': cell['budget'], 'cell_id': cell['cell_id']})
            continue
        split = next(f for f in protocol['split_manifest']['folds'] if f['name'] == cell['fold'])
        ids = split['source_validation_ids'] if cell['fold'] == 'internal' else split['target_development_ids']
        if any(i not in source_ids or source_ids[i]['dataset'] != 'scdd' for i in ids):
            raise PermissionError('No final-confirmation or unaudited identities in development comparisons')
        expected_groups = [source_ids[i]['split_group'] for i in ids]
        records = {}; bindings = []
        for model in cell['models']:
            records[model['family']] = []
            for fit in model['fits']:
                run_path = ROOT / fit['run']; prediction_path = ROOT / fit['predictions']
                if sha256(run_path) != fit['run_sha256'] or sha256(prediction_path) != fit['predictions_sha256']:
                    raise RuntimeError('Bound source run or prediction changed')
                evaluated_path = run_path if cell['fold'] == 'internal' else run_path.parent / 'target_development.json'
                evaluation_binding = next(b for b in cell['identity']['bindings'] if b['run'] == fit['run'])
                if sha256(evaluated_path) != evaluation_binding['evaluation_sha256']:
                    raise RuntimeError('Source native summaries changed')
                run = read_json(run_path)
                with np.load(prediction_path, allow_pickle=False) as saved:
                    arrays = {k: saved[k].copy() for k in ['ids', 'y', 'groups', 'logits']}
                record = saved_record(run, read_json(evaluated_path), arrays, model['family'], ids, expected_groups)
                records[model['family']].append(record)
                for path in {run_path, prediction_path, evaluated_path}:
                    bindings.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
            if sorted(r['model']['seed'] for r in records[model['family']]) != [11, 22, 33]:
                raise ValueError('All3 fitted seeds required before a dense contrast')
        comparisons = []
        for first, second in CONTRASTS:
            if first not in records or second not in records:
                continue
            row = {'first': first, 'second': second, 'endpoints': {}}
            for endpoint in ENDPOINTS:
                try:
                    estimate = paired_endpoint(records[first], records[second], endpoint,
                        repetitions=plan['repetitions'], seed=plan['seed'])
                    row['endpoints'][endpoint] = {'status': 'estimated', 'estimate': estimate}
                except EndpointUnavailable as error:
                    row['endpoints'][endpoint] = {'status': 'not_estimable', 'reason': str(error), 'substitute_endpoint_selected': False}
            comparisons.append(row)
            print('Dense paired development', cell['fold'], cell['budget'], first, '→', second, flush=True)
        result = {'at': now(), 'identity': identity, 'cell_id': cell['cell_id'], 'fold': cell['fold'],
            'budget': cell['budget'], 'images': len(ids), 'positive_images': int(records[next(iter(records))][0]['y'].sum()),
            'provided_capture_families': len(set(expected_groups)),
            'population': cell['evaluation_population'], 'input_bindings': bindings,
            'all_model_seed_points': {family: [{'seed': r['model']['seed'],
                'points': {name: endpoint_value(r, name) for name in ENDPOINTS}} for r in values]
                for family, values in records.items()}, 'comparisons': comparisons,
            'limits': ['Retrospective exploratory development, not a multiplicity-adjusted hypothesis test.',
                'Intervals condition on fitted seeds; source-validation images selected checkpoints.',
                'Provided capture families are conservative proxies, not verified physical modules.',
                'FPR is a ratio of annotated structure counts on crack-negative images, resampled by image/family, never by pixel.',
                'A lower review fraction is a tradeoff that must be interpreted together with recall.',
                'Pooled native AP is a whole-cohort point only, distinct from image-mean AP.'],
            'confirmation_model_scores_opened': False}
        save_json(result_path, result)
        outputs.append({'path': str(result_path.relative_to(ROOT)), 'sha256': sha256(result_path),
                        'fold': cell['fold'], 'budget': cell['budget'], 'cell_id': cell['cell_id']})
    save_json(OUTPUT / 'index.json', {'at': now(), 'plan_sha256': sha256(OUTPUT / 'plan.json'),
        'complete_cells': outputs, 'planned_cells': aggregate['all_planned_cells'],
        'scope': 'Each completed cell has every applicable contrast/endpoint and all3 seeds; partial cells remain omitted. No final confirmation access.'})
    print('Saved conditional dense development comparisons for', len(outputs), 'complete cells', flush=True)


if __name__ == '__main__':
    os.nice(10)
    main()
