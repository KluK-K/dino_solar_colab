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
                if c['fold'] == 'holdout_CSIR' and c['budget'] == .2)
    path = ROOT / cell['path']
    assert sha256(path) == cell['sha256'] == '510992d15082cd724800af6854a314e67ec71edb2566e154ed83096abf4892db'
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
        assert row['training_masks'] == 49 and row['validation_masks'] == 62
        assert row['unlabelled_source_images'] == (198 if 'published_psmt_ssl/' in row['system'] else 0)
        assert (row['images'], row['positive_images'], row['provided_capture_families']) == (
            (62, 29, 31) if source else (162, 66, 103))
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
            'Held LoRA AUROC0.891204 exceeds adaptedCNN0.829388 in all3paired seeds, but nativeAP0.081874 and positiveDice0.123297 are below CNN0.206973/0.189533 in every pair. LoRA mean structureFPR0.001634 exceeds CNN0.000730 with2higher/1lower paired ratios. No uniform localization superiority.',
            'LoRA held recall0.994949/review0.763374 versus CNN0.868687/0.648148 represents different achieved sensitivities. LoRA empty positives14/26/5 and zero-overlap16/34/9 remain out of66. CNN empty26/30/19 and zero-overlap30/34/23. Fewer empty positives in all3LoRA fits coexist with lower mean Dice in all3pairs; a nonempty mask alone does not establish better overlap.',
            'Held supervised local port AUROC0.575021/AP0.070549/Dice0.001425; teacher1 SSL0.558133/0.046271/0.000703 and ensemble0.554819/0.044803/0.000650. SSLprimary minus supervised held AP and Dice adverse all3; AUROC2adverse/1favorable. Lower SSLreview0.759259 versus FSL0.855967 accompanies lower achieved recall0.767677 versus0.853535, not matched-recall efficiency.',
            'Supervised held positive-empty63/62/63 and zero-overlap63/65/64; all-image empty159/158/159 of162; remaining predicted pixels1335/527/1279 are retained, with0negative-field pixels in all3. SSL teacher/ensemble empty-positive65/66/63, zero-overlap65/66/64 and all-image-empty161/162/159; seed22 is entirely empty but seed11/33 sparse recoveries remain. Their0normal-structure FPR is not evidence of adequate crack sensitivity.',
            'All3DINO systems have higher held Dice than each of the3port views in every fitted pair. Frozen/reference mean held AUROC exceed all port means but each comparison has only1positive/2negative paired AUROC directions. LoRA held AP mean exceeds supervised port, yet only1positive/2negative fitted pairs; frozen has2positive/1negative and reference1positive/2negative. Every DINO held AP exceeds SSLprimary/ensemble in all3pairs.',
            'Every DINO system has lower source nativeAP than every applicable port view in all3pairs, despite higher Dice in every pair. Source LoRA AUROC exceeds all ports in all3pairs. Source frozen/reference AUROC mean advantages over SSL have1favorable/2adverse directions. SSLprimary source AUROC improves versus supervised in2pairs, but source Dice mean falls despite2positive/1negative differences. Endpoint means do not establish seed-stable superiority.',
            'Held ensemble AUROC and AP are lower than primary in every pair; Dice lower2/tied1, recall lower1/tied2. Source ensemble AUROC1positive/1negative/1tie, AP1positive/2negative, Dice lowerall3; preserve primary/ensemble correlated views without selecting the favorable view after observation.',
            'LoRA held normal-structureFPR0for seed22 coexists with3080negative-field predicted pixels; seed11/33 negative pixels34162/51107. AdaptedCNN source structureFPR0all3 coexists with120/189/210negative-field pixels. Eligible structure subsets, whole-negative fields, image recall/review and crack overlap are separate outcomes.',
            'All30original contrasts/six endpoints/3paired seeds and42native failure rows retained.49training masks PLUS62validation masks; SSL198unlabelled source-training images; UCF574upstream exposure additional. Same already observed CSIR162/66positive/103provided-family cohort as100%, source62/29/31; no new domain replication, physical-module identity, matching-specific causal effect, confidence interval or adjusted significance.'],
        'annotation_exposure': {'training_masks': 49, 'validation_masks': 62,
            'SSL_unlabelled_source_training_images': 198, 'additional_upstream_UCF_exposure': 574},
        'new_model_fit_inference_or_bootstrap': False, 'independent_confirmation': False,
        'scientific_sources_protocols_checkpoints_unchanged': True}
    assert read_json(ROOT / 'artifacts/publication_20261017/inference_efficiency_v1/status.json')['state'].startswith('queued'), 'Do not compete with formal efficiency'
    save_json(target, review)
    print({'review': str(target.relative_to(ROOT)), 'sha256': sha256(target),
           'reviewed_endpoint_summaries': 180, 'native_failure_rows': 42})


if __name__ == '__main__':
    main()
