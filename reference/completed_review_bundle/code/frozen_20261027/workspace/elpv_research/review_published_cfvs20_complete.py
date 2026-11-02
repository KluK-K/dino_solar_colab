"""Append-only editorial review of an already completed, registered cell.

Read saved summaries only: no fitting, model construction, scoring or bootstrap.
"""
import math
import statistics
from pathlib import Path

from elresearch.common import ROOT, now, read_json, save_json, sha256

OUT = ROOT / 'artifacts/publication_20261017/published_complete_development_readout_v1'


def main():
    index = read_json(OUT / 'index.json')
    cell = next(c for c in index['complete_cells']
                if c['fold'] == 'holdout_CFVS' and c['budget'] == .2)
    path = ROOT / cell['path']
    assert sha256(path) == cell['sha256'] == '14fa09eded3b7b087d6919c34c05959a190988180ee0868c9040352488971472'
    result = read_json(path)
    target = path.parent / 'numerical_review.json'
    assert not target.exists(), 'Do not overwrite a completed review'
    assert result['unique_fits'] == 18
    assert result['correlated_fit_view_population_records'] == 42
    assert result['paired_model_population_contrasts'] == 30
    assert result['endpoint_summaries'] == 180
    assert sha256(ROOT / cell['outputs']['path']) == cell['outputs']['sha256']
    for binding in read_json(ROOT / cell['outputs']['path'])['files']:
        assert sha256(ROOT / binding['path']) == binding['sha256']
    for binding in result['input_bindings']:
        assert sha256(ROOT / binding['path']) == binding['sha256']

    def close(a, b):
        assert (a is None and b is None) or (
            a is not None and b is not None and
            math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)), (a, b)

    rows = result['all_seed_records']
    for population, systems in result['systems'].items():
        assert len(systems) == 7
        for system, endpoints in systems.items():
            fitted = sorted((r for r in rows if r['population'] == population
                             and r['system'] == system), key=lambda r: r['seed'])
            assert [r['seed'] for r in fitted] == [11, 22, 33]
            for endpoint, summary in endpoints.items():
                values = [r[endpoint] for r in fitted]
                assert values == summary['values']
                available = [v for v in values if v is not None]
                assert summary['available'] == len(available)
                close(summary['mean'], statistics.mean(available) if available else None)
                close(summary['sample_SD'], statistics.stdev(available) if len(available) > 1 else None)
                assert not summary['SD_is_confidence_interval']

    signs = []
    for contrast in result['comparisons']:
        entry = {k: contrast[k] for k in ['population', 'first', 'second']}
        entry['endpoints'] = {}
        for endpoint, values in contrast['endpoints'].items():
            a, b = values['first_values'], values['second_values']
            assert a == result['systems'][contrast['population']][contrast['first']][endpoint]['values']
            assert b == result['systems'][contrast['population']][contrast['second']][endpoint]['values']
            differences = [x-y if x is not None and y is not None else None for x, y in zip(a, b)]
            summary = values['first_minus_second']
            assert differences == summary['values'] and len(differences) == 3
            available = [v for v in differences if v is not None]
            close(summary['mean'], statistics.mean(available) if available else None)
            close(summary['sample_SD'], statistics.stdev(available) if len(available) > 1 else None)
            entry['endpoints'][endpoint] = {
                'mean': summary['mean'], 'positive': sum(v > 0 for v in available),
                'negative': sum(v < 0 for v in available), 'zero': sum(v == 0 for v in available),
                'null': 3-len(available)}
        assert len(entry['endpoints']) == 6
        signs.append(entry)
    assert len(signs) == 30

    native = []
    for row in rows:
        run_path = ROOT / row['run']
        run = read_json(run_path)
        source = row['population'] == 'source_validation'
        evaluated = run if source else read_json(run_path.parent / 'target_development.json')
        if '/' in row['system']:
            evaluated = evaluated['inference_views'][row['system'].split('/')[-1]]
        local = evaluated['validation_localization' if source else 'localization']['per_image']
        positive = [v for v in local if v['crack_positive']]
        assert len(local) == row['images'] and len(positive) == row['positive_images']
        assert sum(v['predicted_crack_pixels'] == 0 for v in positive) == row['empty_positive_images']
        assert sum(v['predicted_crack_pixels'] == v['false_positive_pixels'] for v in positive) == row['zero_overlap_positive_images']
        assert row['training_masks'] == 50 and row['validation_masks'] == 64
        assert row['unlabelled_source_images'] == (199 if 'published_psmt_ssl/' in row['system'] else 0)
        assert (row['images'], row['positive_images'], row['provided_capture_families']) == (
            (64, 23, 37) if source else (158, 99, 92))
        native.append({**{k: row[k] for k in ['population', 'system', 'seed', 'images',
                        'positive_images', 'empty_positive_images', 'zero_overlap_positive_images']},
                       'all_images_empty': sum(v['predicted_crack_pixels'] == 0 for v in local),
                       'total_predicted_crack_pixels': sum(v['predicted_crack_pixels'] for v in local),
                       'predicted_crack_pixels_on_negative_images': sum(
                           v['predicted_crack_pixels'] for v in local if not v['crack_positive'])})
    assert len(native) == 42
    review = {
        'at': now(), 'result': cell,
        'review_source': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha256(Path(__file__))},
        'all_applicable_registered_models_and_three_seeds_retained': True,
        'unique_fits': 18, 'correlated_fit_population_records': 42,
        'paired_model_population_contrasts': 30,
        'all_endpoint_summaries_numerically_reviewed': 180, 'native_failure_rows_reviewed': 42,
        'whole_fixed_cohort_empty_mask_diagnostic': native, 'all_contrast_sign_counts': signs,
        'diagnostic_scope': 'Every original source and held image at unchanged0.5 native cutoff; derived aggregate summaries only. No new endpoint, inference, bootstrap, threshold or example selection.',
        'findings': [
            'Held LoRA AUROC0.9823 and Dice0.1829 exceed adapted CNN0.6238/0.1058 in all3 seed directions, but nativeAP0.1490 is below CNN0.1795 (1 favorable/2 adverse paired AP differences). CNN seed33 has all source and held masks empty; retain it.',
            'Held LoRA recall0.9966 requires review0.9093; CNN0.9933/0.9873. These are source-frozen points at different achieved target sensitivities, not matched-recall cost comparisons.',
            'Held LoRA positive-empty counts4/3/8 and zero-overlap4/5/10 remain. Eligible structureFPR0 all3 does not exclude291 negative-field pixels in seed33.',
            'Held supervised port empty-positive counts97/95/99; first2 seeds recover some positive pixels, so do not describe all3 as completely empty. Held nativeAP0.0992 and Dice0.00119; seed33 AUROC0.2708 remains unflipped.',
            'Held SSL teacher1 and ensemble each have empty masks on all158 images/all3 seeds. Teacher1 imageAUROC0.7388 exceeds supervised0.4911 in all3 pairs; its nativeAP0.1259 gain has2 favorable/1 adverse pairs, while Dice is lower in2 pairs and tied in1.',
            'Held ensemble AUROC mean0.7029 is below teacher1 mean0.7388 despite2 favorable/1 adverse paired directions; the larger seed33 loss controls the mean. NativeAP is adverse2/favorable1. Source ensemble seed22 predicts6 pixels on one positive image; source primary masks are all empty.',
            'Reference seed22 has all source and held native masks empty. Its held nativeAP versus SSLprimary is adverse2/favorable1; no universal reference benefit.',
            'All30 contrasts/six endpoints and every fitted seed remain. Descriptive means/sampleSDs are not confidence intervals or adjusted significance. Same CFVS158 cohort as100%, overlapping source data and additional UCF574 exposure; no independent domain replication or matching-specific causal claim.'],
        'new_model_fit_inference_or_bootstrap': False, 'independent_confirmation': False,
        'scientific_sources_protocols_checkpoints_unchanged': True}
    save_json(target, review)
    print({'review': str(target.relative_to(ROOT)), 'sha256': sha256(target),
           'reviewed_endpoint_summaries': 180, 'native_failure_rows': 42})


if __name__ == '__main__':
    main()
