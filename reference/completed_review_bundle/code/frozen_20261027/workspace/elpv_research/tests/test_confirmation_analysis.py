"""Synthetic scored outputs only; verify inference and uncertainty semantics."""
import copy
import numpy as np
import pytest

from elresearch.confirmation_analysis import endpoint_value, paired_endpoint, pair_records
from elresearch.publication_statistics import paired_seed_group_bootstrap


def record(seed=11, scores=None):
    y = np.tile([0, 1], 6); ids = np.array([f'fixture{i}' for i in range(12)])
    logits = np.tile([-.4, .4], 6) + np.repeat(np.linspace(-.5, .5, 6), 2)
    return {'ids': ids, 'y': y, 'groups': np.repeat(np.arange(6), 2),
        'logits': logits if scores is None else np.array(scores), 'source_threshold': -.1,
        'analysis_stratum': 'whole', 'model': {'id': f'fixture_seed{seed}', 'seed': seed,
            'training_fraction': .2, 'kind': 'dense'},
        'localization': {'pooled_native_pixel_AP': .7, 'box_per_image': [
            {'id': ids[1], 'maximum_point_in_target_box': True},
            {'id': ids[3], 'maximum_point_in_target_box': False}],
            'native_pixel_per_image': [
                {'id': i, 'crack_positive': bool(target), 'dice': .6 if target else 1.,
                    'iou': .4 if target else 1., 'negative_image_normal_structure_pixels': 10 if not target else None,
                    'negative_image_normal_structure_false_positive_pixels': 2 if not target else None}
                for i, target in zip(ids, y)]}}


def test_image_intervals_match_original_registered_pairing_and_strict_ties():
    first = [record(s) for s in [11, 22, 33]]
    second = [{**r, 'logits': r['logits'] + np.linspace(-.3, .2, 12)} for r in first]
    convert = lambda rows: [{**r, 'seed': r['model']['seed'], 'threshold': r['source_threshold']} for r in rows]
    original = paired_seed_group_bootstrap(convert(first), convert(second), repetitions=50)
    mapping = {'AUROC': 'auroc', 'average_precision': 'average_precision',
        'recall_at_source_threshold': 'recall_at_source_threshold',
        'review_rate_at_source_threshold': 'review_rate_at_source_threshold',
        'false_positive_rate_at_source_threshold': 'false_positive_rate_at_source_threshold'}
    for endpoint, name in mapping.items():
        actual = paired_endpoint(first, second, endpoint, repetitions=50)
        expected = original['metrics'][name]
        assert actual['mean_pair_difference'] == expected['mean_seed_difference']
        np.testing.assert_allclose(actual['acquisition_family_percentile_ci95'], expected['acquisition_family_percentile_ci95'], atol=1e-14)
    tied = record(scores=np.full(12, -.1))
    assert endpoint_value(tied, 'known_crack_recall') == 0
    assert endpoint_value(tied, 'author_normal_FPR') == 0


def test_native_endpoints_exclude_empty_mask_true_negative_dice():
    row = record()
    assert endpoint_value(row, 'positive_image_Dice') == pytest.approx(.6)
    assert endpoint_value(row, 'positive_image_IoU') == pytest.approx(.4)
    assert endpoint_value(row, 'normal_structure_pixel_FPR') == pytest.approx(.2)
    assert endpoint_value(row, 'box_point_hit_rate') == .5
    row['localization']['native_pixel_per_image'][0].update(
        negative_image_normal_structure_pixels=100, negative_image_normal_structure_false_positive_pixels=0)
    assert endpoint_value(row, 'normal_structure_pixel_FPR') == pytest.approx(10/150)
    for endpoint in ['positive_image_Dice', 'positive_image_IoU', 'normal_structure_pixel_FPR', 'box_point_hit_rate']:
        actual = paired_endpoint([row], [row], endpoint, repetitions=50)
        assert actual['mean_pair_difference'] == 0
        assert actual['acquisition_family_percentile_ci95'] == [0., 0.]


def test_pooled_ap_never_substitutes_per_image_ap_and_missing_pvel_groups_have_no_ci():
    a = record(); b = copy.deepcopy(a); b['localization']['pooled_native_pixel_AP'] = .8
    actual = paired_endpoint([a], [b], 'pooled_native_pixel_AP')
    assert actual['mean_pair_difference'] == pytest.approx(.1)
    assert actual['acquisition_family_percentile_ci95'] is None
    with pytest.raises(ValueError, match='whole-cohort'):
        endpoint_value({**a, 'analysis_stratum': 'ARTS'}, 'pooled_native_pixel_AP')
    missing = {**a, 'groups': np.array([None]*12)}
    for endpoint in ['known_crack_recall', 'author_normal_FPR', 'box_point_hit_rate']:
        result = paired_endpoint([missing], [missing], endpoint, acquisition_groups_available=False)
        assert result['acquisition_family_percentile_ci95'] is None
        assert result['mean_pair_difference'] == 0
    with pytest.raises(ValueError, match='Missing acquisition'):
        paired_endpoint([missing], [missing], 'AUROC')


def test_single_deterministic_fit_is_shared_without_invented_training_seeds():
    deterministic = record(); deterministic['model'].update(kind='classical', training_fraction=1.)
    stochastic = [record(s) for s in [11, 22, 33]]
    result = paired_endpoint([deterministic], stochastic, 'AUROC', repetitions=50)
    assert result['first_unique_fitted_predictors'] == 1
    assert result['second_unique_fitted_predictors'] == 3
    assert [p['first_seed'] for p in result['fitted_pairs']] == [11, 11, 11]
    assert [p['second_seed'] for p in result['fitted_pairs']] == [11, 22, 33]
    assert result['acquisition_family_percentile_ci95'] == [0., 0.]
    with pytest.raises(ValueError, match='Unpaired stochastic'):
        pair_records([record()], stochastic)
    with pytest.raises(ValueError, match='aligned'):
        pair_records([record()], [{**record(), 'ids': record()['ids'][::-1]}])


def test_unsupported_prespecified_endpoint_fails_without_choosing_replacement():
    missing = record(); missing['localization']['native_pixel_per_image'] = []
    with pytest.raises(ValueError, match='annotation/class support'):
        paired_endpoint([missing], [missing], 'positive_image_Dice')
    with pytest.raises(ValueError, match='Unknown'):
        endpoint_value(record(), 'best-looking-map')


def test_complete_saved_prediction_analysis_is_resumable_and_keeps_unestimable_contrasts(tmp_path, monkeypatch):
    from elresearch import confirmation_registry as gate
    from elresearch import confirmation_workflow as workflow
    from elresearch.confirmation_analysis import analyze
    from elresearch.common import save_json, sha256
    from elresearch.metrics import classification
    from elresearch.publication_train import probability_metrics
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    cohort = 'scdd_publication_v1'; folder = tmp_path/'final'; folder.mkdir()
    data = record(); rows = [{'id': str(i), 'partition': 'test', 'target': int(y),
        'split_group': str(g), 'acquisition_source': 'ARTS'}
        for i, y, g in zip(data['ids'], data['y'], data['groups'])]
    manifest = tmp_path/'manifest.json'; save_json(manifest, {'records': rows})
    statistics = tmp_path/'statistics.json'; save_json(statistics, {'repetitions': 50, 'seed': 20261017})
    models = [{**data['model'], 'id': name+'_seed11', 'family': name,
        'source_threshold': -.1, 'source_temperature': 2.} for name in ['a', 'b']]
    jobs = [{'model_id': m['id'], 'cohort': cohort, 'analysis_stratum': 'whole', 'analysis_role': 'synthetic fixture'} for m in models]
    declaration = {'id': 'synthetic-auroc', 'first_family': 'a', 'second_family': 'b',
        'cohort': cohort, 'analysis_stratum': 'whole', 'matched_annotation_identities': True, 'endpoint': 'AUROC'}
    registry = {'models': models, 'cohorts': [{'name': cohort, 'partition': 'test', 'manifest': gate.bind_file(manifest)}],
        'evaluator_files': [], 'analysis_plan': {'confirmation_jobs': jobs,
            'primary_comparisons': [declaration, {**declaration, 'id': 'synthetic-no-masks', 'endpoint': 'positive_image_Dice'}],
            'statistics_plan_binding': gate.bind_file(statistics)}}
    registry['registry_id'] = gate.registry_digest(registry); save_json(folder/'frozen_registry.json', registry)
    audit_jobs = []
    for model, job in zip(models, jobs):
        destination = folder/'results'/model['id']/cohort; destination.mkdir(parents=True)
        np.savez_compressed(destination/'predictions.npz', ids=data['ids'], y=data['y'], logits=data['logits'])
        result = {'registry_id': registry['registry_id'], 'model_id': model['id'], 'cohort': cohort,
            'predictions_sha256': sha256(destination/'predictions.npz'),
            'metrics': classification(data['y'], data['logits'], -.1), 'source_frozen_threshold': -.1,
            'source_frozen_temperature': 2., 'threshold_fitted_on_target': False,
            'source_temperature_probability_metrics': probability_metrics(data['y'], data['logits'], 2.),
            'task_scope': 'Synthetic fixture only', 'development_exposure': {}, 'localization': None}
        save_json(destination/'result.json', result)
        audit_jobs.append({**job, 'result': gate.bind_file(destination/'result.json')})
    monkeypatch.setattr(workflow, 'evaluate', lambda *a: pytest.fail('Analysis must not initiate model inference'))
    save_json(folder/'execution_audit.json', {'registry_id': registry['registry_id'], 'verified_jobs': 1, 'jobs': audit_jobs[:1]})
    with pytest.raises(RuntimeError, match='Every frozen job'): analyze(folder)
    save_json(folder/'execution_audit.json', {'registry_id': registry['registry_id'], 'verified_jobs': 2, 'jobs': audit_jobs})
    proof = analyze(folder)
    assert len(proof['jobs']) == 2 and len(proof['primary_comparisons']) == 2
    assert proof['primary_comparisons'][0]['estimate']['acquisition_family_percentile_ci95'] == [0., 0.]
    assert proof['primary_comparisons'][1]['status'] == 'not_estimable'
    assert not proof['primary_comparisons'][1]['substitute_endpoint_selected']
    assert analyze(folder) == proof
    path = folder/'results'/models[0]['id']/cohort/'predictions.npz'; path.write_bytes(b'changed fixture')
    with pytest.raises(RuntimeError, match='prediction identity'): analyze(folder)
