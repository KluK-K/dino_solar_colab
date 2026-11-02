"""Audit saved matched-UCF DINO controls, without loading a learned model.

This v2 audit explicitly amends a v1 optional-annotation schema assumption.
It does not amend any registered training closure or numerical outcome.
The original fresh-model reload receipts are checked, not rerun. Native pooled
AP and sampled checkpoint-selection AP are not recomputed from raw pixels.
"""
import argparse
import copy
from pathlib import Path
import statistics

import numpy as np

from analyze_dense_development_uncertainty import saved_record
from elresearch.common import ROOT, digest, now, read_json, save_json, sha256
from elresearch.metrics import classification, recall_threshold
from elresearch.publication_train import probability_metrics


BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'published_methods_v1'
OUT = BASE / 'matched_dino_source_output_integrity_v2'
NAMES = ['dinov3_frozen_ucf20', 'dinov3_lora_ucf20', 'dinov3_reference_ucf20']
SEEDS = [11, 22, 33]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(first, second):
    if first is None or second is None:
        require(first is second, 'Null endpoint changed')
    else:
        require(np.isclose(first, second, rtol=0, atol=2e-12), 'Saved endpoint changed')


def guard():
    state = read_json(BASE / 'inference_efficiency_v1/status.json')['state']
    require(state.startswith('queued') or state == 'complete',
            'No output audit during formal efficiency measurements')
    require(not (BASE / 'final_confirmation_v1/observation_opened.json').exists(),
            'Source audit must precede final confirmation')


def binding(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def immutable(path, value):
    if path.exists():
        require(read_json(path) == value, 'Preserve previously bound audit evidence: ' + str(path))
    else:
        save_json(path, value)


def structural_annotation_count(row):
    value = row.get('negative_image_normal_structure_pixels')
    require(value is None or (type(value) is int and value >= 0),
            'Invalid optional normal-structure annotation support')
    return value


def arrays_at(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in ['ids', 'y', 'groups', 'logits']}


def validate_values(run, arrays, history, expected):
    """Check saved numeric values and identities; never construct a network."""
    require(run['study_id'] == expected['study_id'], 'Wrong original study')
    require(run['model'] == expected['model'], 'Wrong registered model specification')
    require(run['scientific_hashes'] == expected['scientific_hashes'], 'Wrong scientific closure')
    require(run['seed'] == expected['seed'], 'Wrong fitted seed')
    require(run['training_budget_fraction'] == expected['budget'], 'Wrong annotation budget')
    require(run['training_ids'] == expected['training_ids'], 'Training identities differ')
    require(run['validation_ids'] == expected['validation_ids'], 'Validation identities differ')
    require(len(set(run['training_ids'])) == len(run['training_ids']), 'Duplicate training identities')
    require(not set(run['training_ids']) & set(run['validation_ids']), 'Source partition overlap')
    require(run['training_images'] == len(run['training_ids']), 'Wrong training exposure')
    require(run['validation_images'] == len(run['validation_ids']), 'Wrong validation exposure')
    require(run['total_pixel_annotated_images_including_validation'] ==
            run['training_images'] + run['validation_images'], 'Wrong total annotation exposure')
    require(len(history) == run['completed_rounds'] and bool(history), 'Incomplete optimizer history')
    require([row['round'] for row in history] == list(range(1, len(history) + 1)),
            'Optimizer rounds are not contiguous')
    require(all(row['optimizer_updates'] == row['round'] * expected['updates_per_round']
                for row in history), 'Optimizer update history differs')
    require(run['optimizer_updates'] == history[-1]['optimizer_updates'], 'Wrong terminal optimizer count')
    require(run['best_round'] == max(history, key=lambda x: x['validation_selection_pixel_AP'])['round'],
            'Original source checkpoint selection differs')
    require(run['first_gradient_check']['heads_have_gradient'], 'Missing decoder gradient receipt')
    require(bool(run['first_gradient_check']['encoder_has_gradient']) ==
            (run['model']['adaptation'] != 'frozen'), 'Encoder gradient contract differs')
    require(np.array_equal(arrays['y'], expected['y']), 'Independent saved source truth differs')
    record = saved_record(run, run, arrays, run['model']['name'],
                          expected['validation_ids'], expected['groups'])
    for key, value in classification(record['y'], record['logits'], run['source_logit_threshold']).items():
        close(value, run['validation_metrics'][key])
    close(recall_threshold(record['y'], record['logits'], expected['target_source_recall']),
          run['source_logit_threshold'])
    for temperature, key in [(1., 'validation_calibration_raw'),
                             (run['source_temperature'], 'validation_calibration_temperature_scaled')]:
        for metric, value in probability_metrics(record['y'], record['logits'], temperature).items():
            close(value, run[key][metric])
    rows = run['validation_localization']['per_image']
    annotation = [(row['id'], row['known_pixels'], row['crack_pixels'],
                   structural_annotation_count(row)) for row in rows]
    require(annotation == expected['annotation'], 'Native source annotation support differs')
    for row in rows:
        for key in ['known_pixels', 'crack_pixels', 'predicted_crack_pixels',
                    'false_positive_pixels', 'known_noncrack_pixels']:
            require(type(row[key]) is int, 'Native confusion counts must be integers')
        require(row['known_noncrack_pixels'] == row['known_pixels'] - row['crack_pixels'],
                'Native noncrack denominator differs')
        true_positive = row['predicted_crack_pixels'] - row['false_positive_pixels']
        union = row['predicted_crack_pixels'] + row['crack_pixels'] - true_positive
        close(true_positive / union if union else 1., row['iou'])
    local = run['validation_localization']
    unprovided_structure = all(structural_annotation_count(row) is None for row in rows)
    if unprovided_structure:
        require(local['negative_images_with_structural_annotation'] == 0 and
                local['pooled_negative_structure_false_positive_fraction'] is None,
                'Unprovided normal-structure endpoint must remain null, not zero')
    positive = [row for row in rows if row['crack_positive']]
    negative = [row for row in rows if not row['crack_positive']]
    require((local['images'], local['positive_images'], local['negative_images']) ==
            (len(rows), len(positive), len(negative)), 'Native population sizes differ')
    close(statistics.mean(row['iou'] for row in positive), local['mean_positive_image_iou'])
    close(statistics.mean(row['dice'] for row in rows), local['mean_dice_all_images'])
    close(statistics.mean(row['negative_image_false_positive_fraction'] for row in negative),
          local['mean_negative_image_false_positive_fraction'])
    reload = run['checkpoint_reload']
    require(reload['fresh_model'] is True, 'Missing original fresh-model reload receipt')
    for key in ['logit_max_abs_error', 'sampled_pixel_probability_max_abs_error']:
        require(np.isfinite(reload[key]) and 0 <= reload[key] <= 2e-5,
                'Original fresh-model reload tolerance failed')
    return {'source_images': len(rows), 'positive_images': len(positive),
            'native_integer_count_records_verified': len(rows),
            'normal_structure_annotation_support': 'not_estimable_unprovided' if unprovided_structure else 'provided',
            'original_normal_structure_FPR_retained': local['pooled_negative_structure_false_positive_fraction'],
            'original_fresh_model_reload_receipt_checked': reload,
            'new_model_reload_or_inference': False, 'native_pooled_AP_recomputed': False,
            'sampled_selection_pixel_AP_recomputed': False}


def registration():
    published = read_json(SOURCE / 'protocol.json')
    dense = read_json(BASE / 'dense_development_v1/protocol.json')
    upstream = read_json(BASE / 'dense_development_v1/selected_hyperparameters.json')['selected']
    frozen = next(row for row in upstream if row['adaptation'] == 'frozen')
    lora = next(row for row in upstream if row['adaptation'] == 'lora')
    models = [{**frozen, 'name': NAMES[0]}, {**lora, 'name': NAMES[1]},
              {**frozen, 'name': NAMES[2], 'reference': True, 'geometry_weight': .1, 'top_fraction': .05}]
    require(sha256(ROOT / 'run_published_methods.py') == published['runner_sha256'],
            'Original published-method runner changed')
    for name, value in published['dense_control_scientific_hashes'].items():
        require(sha256(ROOT / 'elresearch' / name) == value, 'Original dense closure changed: ' + name)
    encoder_reference = (BASE / 'dense_development_v1/runs/internal/budget_0.2/'
                         'dinov3_dense_frozen_seed11/run.json')
    paths = [Path(__file__), ROOT / 'verify_matched_dino_source_outputs.py',
             BASE / 'matched_dino_source_output_integrity_v1/plan.json',
             BASE / 'operational_recovery_v1/selection_metadata_2026-10-27T00-31-35.243048+00-00/audit_schema_amendment_v2/archived_failure_inputs.json', SOURCE / 'protocol.json', BASE / 'dense_development_v1/protocol.json',
             BASE / 'dense_development_v1/selected_hyperparameters.json',
             BASE / 'source_development_identity_v1.json', ROOT / 'run_published_methods.py',
             ROOT / 'analyze_dense_development_uncertainty.py', ROOT / 'elresearch/confirmation_analysis.py',
             encoder_reference]
    paths += [ROOT / 'elresearch' / name for name in published['dense_control_scientific_hashes']]
    plan = {'version': 2,
            'amendment': 'Preserve failed v1 auditor/plan/log. Optional unprovided native normal-structure support is represented by None and its endpoint must remain null, never an imputed zero. Existing provided SCDD support stays exact. No scientific model, score or source selection change.', 'kind': 'Additional saved-output integrity audit for the original nine matched-UCF DINO controls',
            'input_bindings': [binding(path) for path in sorted(set(paths))],
            'models': models, 'seeds': SEEDS, 'planned_fits': 9, 'training_fraction': .2,
            'expected_encoder': read_json(encoder_reference)['encoder'],
            'training_identity_reference': 'Original same-seed matched20 full supervised source fit',
            'validation_identity_reference': 'Original published source protocol and independent saved FSL predictions',
            'native_truth_reference': 'Same-seed full supervised source native integer annotation counts',
            'source_threshold_recomputed_for_audit_only': True,
            'temperature_refitted': False, 'new_model_load_or_inference': False,
            'native_pooled_AP_recomputed': False, 'sampled_selection_pixel_AP_recomputed': False,
            'reload_scope': 'Validate the original training-time fresh-model receipt and saved checkpoint hash; no new reload run.',
            'limits': 'Metadata/saved-output consistency only. No raw-image, raw-mask, target, final-test or prediction-performance reproduction. No complete model comparison or independent superiority claim.'}
    immutable(OUT / 'plan.json', plan)
    return published, dense, models


def fixture_check():
    """Real saved development fixtures and in-memory corruption probes."""
    reports = []
    for family in ['dinov3_dense_frozen', 'dinov3_dense_lora', 'dinov3_dense_reference']:
        path = BASE / 'dense_development_v1/runs/internal/budget_0.2' / f'{family}_seed11/run.json'
        run = read_json(path); arrays = arrays_at(path.parent / 'validation_predictions.npz')
        history = read_json(path.parent / 'history.json')
        expected = {key: copy.deepcopy(run[key]) for key in
                    ['study_id', 'model', 'scientific_hashes', 'seed', 'training_ids', 'validation_ids']}
        expected.update(budget=.2, updates_per_round=50, target_source_recall=.95,
                        y=np.array([int(row['crack_positive']) for row in run['validation_localization']['per_image']]),
                        groups=[row['acquisition_family'] for row in run['validation_localization']['per_image']],
                        annotation=[(row['id'], row['known_pixels'], row['crack_pixels'],
                                     structural_annotation_count(row))
                                    for row in run['validation_localization']['per_image']])
        positive = validate_values(run, arrays, history, expected)
        probes = []
        for kind in ['saved_identity', 'saved_logit', 'native_confusion', 'source_temperature',
                     'training_identity', 'reload_receipt']:
            altered = copy.deepcopy(run); saved = {key: value.copy() for key, value in arrays.items()}
            if kind == 'saved_identity': saved['ids'][0] = saved['ids'][1]
            elif kind == 'saved_logit': saved['logits'][0] += 5.
            elif kind == 'native_confusion': altered['validation_localization']['per_image'][0]['dice'] += .1
            elif kind == 'source_temperature': altered['source_temperature'] *= 1.1
            elif kind == 'training_identity': altered['training_ids'] = altered['training_ids'][1:]
            else: altered['checkpoint_reload']['fresh_model'] = False
            try:
                validate_values(altered, saved, history, expected)
            except ValueError:
                probes.append({'probe': kind, 'rejected': True})
            else:
                raise RuntimeError('Corruption probe incorrectly accepted: ' + kind)
        reports.append({'fixture': binding(path), 'predictions': binding(path.parent / 'validation_predictions.npz'),
                        'history': binding(path.parent / 'history.json'), 'positive_check': positive, 'probes': probes})
    result = {'verifier_sha256': sha256(__file__), 'fixtures': reports, 'positive_checks': 3,
              'corruption_probes_rejected': 18, 'new_model_load_inference_fit_or_raw_pixels': False,
              'scope': 'Existing SCDD development output fixtures, not the future nine matched-UCF fits; no research-performance reproduction.'}
    immutable(OUT / 'fixtures' / digest(result)[:20] / 'verification.json', result)
    print('Saved-output fixtures passed:3 real source fits;18 in-memory corruption probes rejected.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--fixture-check', action='store_true')
    args = parser.parse_args(); guard()
    protocol, dense, models = registration()
    if args.fixture_check:
        fixture_check()
    if args.register_only or args.fixture_check:
        print('Registered additional saved-output audit; original nine matched-UCF fits remain independently required.')
        return
    controls = SOURCE / 'matched_dino_controls.json'
    require(controls.exists(), 'Original source-transfer controls have not yet been emitted by the worker')
    control = read_json(controls)
    require(control['selected'] == models and control['source_dense_selection_sha256'] ==
            sha256(BASE / 'dense_development_v1/selected_hyperparameters.json'), 'Matched controls changed')
    identity = {row['id']: row for row in read_json(BASE / 'source_development_identity_v1.json')['records']}
    expected_ids = protocol['source_validation_ids']; completed = []
    for model in models:
        for seed in SEEDS:
            folder = SOURCE / 'source_transfer' / f"{model['name']}_seed{seed}"
            if not (folder / 'run.json').exists():
                continue
            guard()
            run = read_json(folder / 'run.json')
            fsl_folder = SOURCE / 'source_transfer' / f'published_deeplab_fsl_seed{seed}'
            fsl = read_json(fsl_folder / 'run.json'); fsl_arrays = arrays_at(fsl_folder / 'validation_predictions.npz')
            require(fsl['study_id'] == protocol['study_id'] and fsl['seed'] == seed and
                    fsl['training_budget_fraction'] == .2, 'Wrong same-seed source reference')
            require(fsl['validation_ids'] == expected_ids and fsl_arrays['ids'].tolist() == expected_ids,
                    'Source reference validation support changed')
            require(set(fsl['training_ids']) <= set(protocol['source_training_ids']), 'Unregistered reference training support')
            expected = {'study_id': protocol['study_id'], 'model': model, 'seed': seed, 'budget': .2,
                        'scientific_hashes': protocol['dense_control_scientific_hashes'], 'updates_per_round': 50,
                        'training_ids': fsl['training_ids'], 'validation_ids': expected_ids,
                        'groups': [identity[identifier]['split_group'] for identifier in expected_ids],
                        'target_source_recall': protocol['target_source_recall'], 'y': fsl_arrays['y'],
                        'annotation': [(row['id'], row['known_pixels'], row['crack_pixels'],
                                        structural_annotation_count(row))
                                       for row in fsl['inference_views']['primary']['validation_localization']['per_image']]}
            require(not (folder / 'latest.pt').exists(), 'Completed control retains active optimizer state')
            require(sha256(folder / 'best.pt') == run['checkpoint_sha256'], 'Original control checkpoint changed')
            require(run['encoder'] == read_json(OUT / 'plan.json')['expected_encoder'],
                    'Wrong original DINO encoder/weights/config/input identity')
            started = read_json(folder / 'started.json')['signature']
            require(all(run[key] == value for key, value in started.items()), 'Original fit-start signature changed')
            result = validate_values(run, arrays_at(folder / 'validation_predictions.npz'),
                                     read_json(folder / 'history.json'), expected)
            files = [folder / name for name in ['run.json', 'best.pt', 'started.json', 'history.json', 'validation_predictions.npz']]
            files += [controls, fsl_folder / 'run.json', fsl_folder / 'validation_predictions.npz']
            if model.get('reference'):
                recipe_path = folder / 'normal_bank_recipe.json'; recipe = read_json(recipe_path)
                require(recipe['encoder'] == run['encoder'], 'Normal-bank encoder differs')
                require({row['id'] for row in recipe['rows']} <= set(run['training_ids']),
                        'Normal-bank support is not source training only')
                files.append(recipe_path)
                for row in recipe['rows']:
                    payload = ROOT / recipe['cache_directory'] / (row['cache_key'] + '.pt')
                    require(sha256(payload) == row['payload_sha256'], 'Saved normal-bank payload changed')
                    files.append(payload)
            receipt = {'identity': {'plan_sha256': sha256(OUT / 'plan.json'),
                                    'source_files': [binding(path) for path in sorted(set(files))]},
                       'checks': result, 'new_inference_or_fit': False, 'complete_model_comparison': False}
            destination = OUT / 'receipts' / digest(receipt['identity'])[:20] / 'verification.json'
            immutable(destination, receipt)
            completed.append({'run': str((folder / 'run.json').relative_to(ROOT)), 'verification': binding(destination)})
    index = {'at': now(), 'plan_sha256': sha256(OUT / 'plan.json'), 'completed_fits_verified': len(completed),
             'planned_fits': 9, 'all_nine_saved_output_audits_complete': len(completed) == 9,
             'receipts': completed, 'new_model_load_or_inference': False, 'independent_confirmation': False,
             'complete_model_comparison': False}
    save_json(OUT / 'index.json', index)
    print('Verified matched-DINO source saved outputs:', len(completed), '/9; no new model inference.')


if __name__ == '__main__':
    main()
