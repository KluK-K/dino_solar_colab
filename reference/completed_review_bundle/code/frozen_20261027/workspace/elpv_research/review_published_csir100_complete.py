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
                if c['fold'] == 'holdout_CSIR' and c['budget'] == 1.)
    path = ROOT / cell['path']
    assert sha256(path) == cell['sha256'] == '9973cfe411baf203d822531ac32dc47490bf21fd7beac2c50866c685ae35d0c4'
    result = read_json(path)
    target = path.parent / 'numerical_review.json'
    assert not target.exists(), 'Do not overwrite a completed review'
    assert result['unique_fits'] == 15
    assert result['correlated_fit_view_population_records'] == 30
    assert result['paired_model_population_contrasts'] == 8
    assert result['endpoint_summaries'] == 48
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
        assert len(systems) == 5
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
    assert len(signs) == 8

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
        assert row['training_masks'] == 247 and row['validation_masks'] == 62
        assert row['unlabelled_source_images'] == 0
        assert (row['images'], row['positive_images'], row['provided_capture_families']) == (
            (62, 29, 31) if source else (162, 66, 103))
        native.append({**{k: row[k] for k in ['population', 'system', 'seed', 'images',
                        'positive_images', 'empty_positive_images', 'zero_overlap_positive_images']},
                       'all_images_empty': sum(v['predicted_crack_pixels'] == 0 for v in local),
                       'total_predicted_crack_pixels': sum(v['predicted_crack_pixels'] for v in local),
                       'predicted_crack_pixels_on_negative_images': sum(
                           v['predicted_crack_pixels'] for v in local if not v['crack_positive'])})
    assert len(native) == 30
    review = {
        'at': now(), 'result': cell,
        'review_source': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha256(Path(__file__))},
        'all_applicable_registered_models_and_three_seeds_retained': True,
        'unique_fits': 15, 'correlated_fit_population_records': 30,
        'paired_model_population_contrasts': 8,
        'all_endpoint_summaries_numerically_reviewed': 48, 'native_failure_rows_reviewed': 30,
        'whole_fixed_cohort_empty_mask_diagnostic': native, 'all_contrast_sign_counts': signs,
        'diagnostic_scope': 'Every original source and held image at unchanged0.5 native cutoff; derived aggregate summaries only. No new endpoint, inference, bootstrap, threshold or example selection.',
        'findings': [
            'Held LoRA AUROC0.921612 andDice0.142160 exceed adaptedCNN0.487005/0.076694 in all3paired directions; nativeAP0.070994 is belowCNN0.101088 in all3pairs. Eligible structureFPR0.002747 exceedsCNN0.000206 in all3pairs. Endpoints give different conclusions.',
            'Held supervised full local port AUROC0.614057 exceeds adaptedCNN0.487005 in all3pairs, although adaptedCNN nativeAP0.101088/Dice0.076694 exceed port0.062005/0.007001 in all3pairs. Loss/score/pretraining/optimizer differences prevent causal architecture or original-paper claims.',
            'Port held empty-positive counts62/64/57 and zero-overlap63/65/58 out of66. All-image empty masks155/160/152 out of162, not all-empty; total predicted pixels18179/382/5528 and negative-field802/0/18. Keep every sparse overlap and false positive.',
            'LoRA held positive-empty8/13/9 and zero-overlap9/18/12 remain despite recall1.0/all3seeds. Review0.771605, compared with port recall0.924242/review0.890947 and adaptedCNN0.621212/0.664609. These are different achieved sensitivities, not matched-recall rankings.',
            'All3DINO systems have lower source nativeAP than the full port in every paired seed. AdaptedCNN sourceAP mean0.324669 exceeds port0.305990 but only1positive/2negative paired directions. Source means do not establish stable superiority.',
            'LoRA has higher held mean nativeAP than port0.070994 versus0.062005, but only2positive/1negative pairs. Frozen/ref heldAP mean gains versus port likewise2positive/1negative. All3DINO views have higher heldAUROC/Dice and higher eligible structureFPR than port in every pair; preserve the false-positive tradeoff.',
            'LoRA source eligible structureFPR0 for seed22 does not identify absence of every negative-field pixel for other fits. Held LoRA predicts51736/61260/57403 negative-field pixels; normal-structure subset, whole-field pixel error, image recall and review fraction remain distinct.',
            'All8model/population contrasts/six endpoints and3paired seeds retained. NativeAP point-only; descriptive seedSDs are not confidence intervals or adjusted significance.247training masks PLUS62validation masks, no SSLunlabelled pool in this full-budget condition; upstreamUCF574 exposure additional. Held162/66positives/103providedfamilies is already observed development, not independent confirmation or verified physical modules.'],
        'new_model_fit_inference_or_bootstrap': False, 'independent_confirmation': False,
        'scientific_sources_protocols_checkpoints_unchanged': True}
    save_json(target, review)
    print({'review': str(target.relative_to(ROOT)), 'sha256': sha256(target),
           'reviewed_endpoint_summaries': 48, 'native_failure_rows': 30})


if __name__ == '__main__':
    main()
