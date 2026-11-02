"""Audit the complete saved cost readout and append interpretation, not new timing."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
FOLDER = BASE / 'complete_efficiency_readout_v1/4e1f4c8dcffe5ff1c86f'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ref(p):
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def immutable(p, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists(): assert p.read_text() == text
    else: p.write_text(text)


def main():
    results = read(FOLDER / 'results.json'); plan = read(ROOT / results['plan']['path'])
    assert sha(ROOT / results['plan']['path']) == results['plan']['sha256']
    for b in [*plan['bindings'], *read(FOLDER / 'outputs.json')['files']]:
        assert sha(ROOT / b['path']) == b['sha256']
    raw = {}
    for b in plan['result_bindings']:
        p = ROOT / b['result_path']; assert sha(p) == b['result_sha256']
        raw[b['id']] = read(p)
    fit_rows = list(csv.DictReader((FOLDER / 'all_fit_conditions.csv').open()))
    pass_rows = list(csv.DictReader((FOLDER / 'all_primary_passes.csv').open()))
    paired_rows = list(csv.DictReader((FOLDER / 'all_paired_metric_records.csv').open()))
    assert (len(raw), len(fit_rows), len(pass_rows), len(paired_rows)) == (97, 388, 1164, 2112)
    by_case = {(x['model_id'], x['cohort'], x['batch_size']): x for x in results['cases']}
    assert len(by_case) == 388
    for case in results['cases']:
        original = next(c for c in raw[case['model_id']]['cohorts'][case['cohort']] if c['batch_size'] == case['batch_size'])
        repeats = original['repeated_uninstrumented_passes']
        times = sorted(r['timed_pipeline_seconds'] * 1000 / r['images'] for r in repeats)
        same(case['median_pipeline_ms_per_image'], times[1])
        same(case['median_loop_ms_per_image'], sorted(r['observed_loop_seconds'] * 1000 / r['images'] for r in repeats)[1])
        same(case['median_images_per_second'], sorted(r['images'] / r['timed_pipeline_seconds'] for r in repeats)[1])
        for key, value in original['memory']['maximum_observed'].items(): assert case[key] == value
        assert case['component_pass'] == original['component_pass']
        assert case['prediction_equivalence'] == original['prediction_equivalence']
    for row in fit_rows:
        case = by_case[(row['model_id'], row['cohort'], int(row['batch_size']))]
        for metric in plan['metrics']: same(float(row[metric]), case[metric])
    for row in pass_rows:
        source = raw[row['model_id']]
        case = next(c for c in source['cohorts'][row['cohort']] if c['batch_size'] == int(row['batch_size']))
        repeat = case['repeated_uninstrumented_passes'][int(row['repeat']) - 1]
        for key in ['timed_pipeline_seconds', 'observed_loop_seconds', 'weighted_mean_ms_per_image', 'pipeline_images_per_second']:
            same(float(row[key]), repeat[key])
    null_ratios = 0
    for row in paired_rows:
        suffix = '_seed' + row['seed']; batch = int(row['batch_size'])
        a = by_case[(row['first'] + suffix, row['cohort'], batch)][row['metric']]
        b = by_case[(row['second'] + suffix, row['cohort'], batch)][row['metric']]
        same(float(row['first_value']), a); same(float(row['second_value']), b)
        same(float(row['difference']), a - b)
        if b == 0:
            assert row['ratio'] == ''; null_ratios += 1
        else: same(float(row['ratio']), a / b)
    for group in results['family_summaries']:
        cases = [c for c in results['cases'] if (c['family'], c['cohort'], c['batch_size']) == (group['family'], group['cohort'], group['batch_size'])]
        assert sorted(c['seed'] for c in cases) == group['seeds']
        for metric, data in group['metrics'].items():
            values = [c[metric] for c in cases]
            same(data['mean'], sum(values) / len(values)); same(data['median'], st.median(values))
            assert data['minimum'] == min(values) and data['maximum'] == max(values)
            if len(values) > 1: same(data['sample_SD'], st.stdev(values))
            else: assert data['sample_SD'] is None
    original_html = (FOLDER / 'report.html').read_text()
    for family in plan['families']: assert original_html.count(f'<td>{family} / ') == 4
    # The original table reports local source-run metadata. This companion makes
    # inherited UCF labels/pretraining and all measurement caveats explicit.
    clarification = '<aside><h2>Interpretation and reporting scope</h2><p>Training / validation / unlabeled image counts in the table below are the original local source-run metadata, not total study-wide annotation exposure or pretraining cost. Additional upstream UCF574 labels and HPO exposure, pretrained-image exposure and historical preparation are not counted by that table. No label-efficiency claim may be inferred from these counts alone.</p><p>Primary repeats and fitted-seed variability describe this one host and measurement period. Other desktop activity, condition order, thermal state, implementation details and filesystem warmth may affect timings; matching seed numbers does not create synchronized hardware replications. All three latency pass values and every original job are retained. RSS highwater can carry over from earlier conditions in the same process. Do not add overlapping unified-memory pools, call MPS observations guaranteed peaks, or infer training memory/energy from these inference measurements.</p><p>Feature-extraction, classification and remaining-forward component times are nested under model_forward_and_transfers. Adding both levels double-counts work. Component passes are separately instrumented; primary latency uses uninstrumented passes.</p></aside>'
    reviewed = original_html.replace('<h1>All 97 formal source inference-cost jobs</h1>', '<h1>All 97 formal source inference-cost jobs</h1>' + clarification)
    path = FOLDER / 'reviewed_report.html'
    if path.exists(): assert path.read_text() == reviewed
    else: path.write_text(reviewed)
    selected = {}
    for group in results['family_summaries']:
        if group['family'] in ['image__dinov3_lora', 'image__dinov3_full', 'image__resnet50_full', 'dense__dinov3_dense_lora', 'dense__deeplabv3plus_full', 'dense__dinov3_dense_reference', 'dense__dinov3_dense_frozen', 'published__ssl_primary', 'published__ssl_ensemble']:
            selected.setdefault(group['family'], []).append(group)
    review = {'reporting_source': ref(Path(__file__).absolute()), 'result': ref(FOLDER / 'results.json'),
              'original_outputs': ref(FOLDER / 'outputs.json'), 'reviewed_report': ref(path),
              'checked_jobs': 97, 'checked_fit_conditions': 388, 'checked_primary_passes': 1164,
              'checked_family_conditions': len(results['family_summaries']), 'checked_paired_metric_records': 2112,
              'zero_denominator_ratios_retained_null': null_ratios,
              'descriptive_examples_with_full_seed_values': selected,
              'findings': [
                  'DINOv3 LoRA does not establish an inference-speed advantage over the adapted CNN. On SCDD/batch1, dense LoRA mean of per-fit median latency is31.73ms/image versus27.08ms for adapted DeepLab; same-seed ratios vary approximately0.97..1.43. All four cohort/batch conditions have higher mean LoRA latency, but individual seed differences are mixed.',
                  'Dense LoRA observed RSS highwater is lower than adapted DeepLab in these runs. This is one overlapping host memory counter, not training memory, disjoint total usage or a guaranteed peak.',
                  'The normal-reference implementation is slower than frozen dense in all12 same-seed/cohort/batch latency comparisons. It also retains additional reference objects; the complete table reports each source bank size and its tensor/serialized bytes.',
                  'The published SSL ensemble is slower than its primary inference view in all12 same-seed/cohort/batch comparisons; both views remain reported. RSS need not increase monotonically across separate measured processes.',
                  'Full-budget versus20%-budget reference latency is not universally monotonic. Bank size alone does not isolate hardware timing, and these warm-host observations cannot identify a causal budget effect.',
                  'Fewer fitted LoRA parameters do not imply fewer loaded inference parameters, lower latency, lower overall memory, better segmentation or independent superiority.'],
              'new_models_fits_inference_or_bootstrap': False, 'HTML_layout_actually_inspected': False,
              'figures_and_native_manuscript_integration_pending': True,
              'independent_confirmation': False, 'submission_ready': False}
    # Verify the direction claims against every applicable saved paired record.
    for first, second in [('dense__dinov3_dense_reference', 'dense__dinov3_dense_frozen'), ('published__ssl_ensemble', 'published__ssl_primary')]:
        comparisons = [c for c in results['paired_contrasts'] if (c['first'], c['second']) == (first, second)]
        values = [v for c in comparisons for v in c['metrics']['median_pipeline_ms_per_image']['difference']['values']]
        assert len(values) == 12 and min(values) > 0
    immutable(FOLDER / 'numerical_review.json', review)
    print(json.dumps({'review': ref(FOLDER / 'numerical_review.json'), 'checked_fit_conditions': 388, 'checked_primary_passes': 1164, 'checked_paired_metric_records': 2112}))


if __name__ == '__main__':
    main()
