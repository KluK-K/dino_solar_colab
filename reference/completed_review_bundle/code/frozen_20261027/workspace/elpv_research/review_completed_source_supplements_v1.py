"""Complete numerical review of original ELPV and matched UCF source outputs."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
DIRECTORIES = [BASE / 'complete_ELPV_supplement_readout_v1/268150aa6a1f25e7196e',
               BASE / 'matched_UCF_source_readout_v1/e7742c7573c4cb2022a9']


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
def close(a, b): assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def immutable(p, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists(): assert p.read_text() == text
    else: p.write_text(text)


def main():
    for folder in DIRECTORIES:
        result = read(folder / 'results.json'); plan = read(ROOT / result['plan']['path'])
        assert sha(ROOT / result['plan']['path']) == result['plan']['sha256']
        for binding in [*plan['bindings'], *read(folder / 'outputs.json')['files']]:
            assert sha(ROOT / binding['path']) == binding['sha256']
    ep, uc = [read(p / 'results.json') for p in DIRECTORIES]
    ep_points = {(p['feature'], p['classifier'], p['outer_fold']): p for p in ep['per_fit']}
    csv_ep = list(csv.DictReader((DIRECTORIES[0] / 'all_outer_fold_metrics.csv').open())); assert len(csv_ep) == 192
    for row in csv_ep:
        close(float(row['value']), ep_points[(row['feature'], row['classifier'], int(row['outer_fold']))]['metrics_at_default_zero'][row['metric']])
    ep_pairs = list(csv.DictReader((DIRECTORIES[0] / 'all_paired_feature_differences.csv').open())); assert len(ep_pairs) == 96
    for row in ep_pairs:
        a = ep_points[('dinov3_cls', row['classifier'], int(row['outer_fold']))]['metrics_at_default_zero'][row['metric']]
        b = ep_points[('hog_stats', row['classifier'], int(row['outer_fold']))]['metrics_at_default_zero'][row['metric']]
        close(float(row['difference']), a - b)
    for group in ep['model_summaries']:
        for metric, summary in group['metrics'].items():
            values = [ep_points[(group['feature'], group['classifier'], fold)]['metrics_at_default_zero'][metric] for fold in [0, 1, 2, 3]]
            assert values == summary['values']; close(summary['mean'], st.mean(values)); close(summary['sample_SD'], st.stdev(values))
    auc_gaps = [value for contrast in ep['paired_feature_contrasts'] if contrast['metric'] == 'auc' for value in contrast['summary']['values']]
    assert len(auc_gaps) == 12 and min(auc_gaps) > 0
    ep_review = {'review_code': ref(Path(__file__).absolute()), 'result': ref(DIRECTORIES[0] / 'results.json'),
                 'checked_outer_fits': 24, 'checked_saved_metric_values': 192, 'checked_same_fold_endpoint_differences': 96,
                 'all_classifiers_and_folds_reported': True, 'all12same_classifier_AUROC_differences_positive': True,
                 'findings': ['Frozen DINOv3 CLS improves observed review-label outer AUROC over HOG/statistics in all4folds for each original classifier. The mean RBF values are0.904918 versus0.838261, with all4paired differences retained.',
                              'All8original endpoints and all6systems are retained. RBF recall increases with a higher mean review fraction; lower review alone is not a sensitivity-matched workload ranking.',
                              'The four outer training sets overlap. These are1968reused expert-review-label development images, not crack masks, independent module populations, CNN controls or a reproduction of published Deitsch scores. The656course-test images remain unscored.',
                              'The480inner-grid count comes from successful unchanged grids; no per-inner checkpoints/full CV score tables were saved. No claim of480independently audited checkpoints is made.'],
                 'new_model_fit_inference_or_bootstrap': False, 'independent_confirmation': False,
                 'figure_QA_and_native_article_integration_separate': True}
    immutable(DIRECTORIES[0] / 'numerical_review.json', ep_review)
    uc_points = {(p['system'], p['seed']): p for p in uc['per_fit']}
    csv_uc = list(csv.DictReader((DIRECTORIES[1] / 'all_fit_endpoints.csv').open())); assert len(csv_uc) == 12
    endpoint_values = 0
    for row in csv_uc:
        point = uc_points[(row['system'], int(row['seed']))]
        for metric, value in point['metrics'].items():
            endpoint_values += 1
            if value is None: assert row[metric] == ''
            else: close(float(row[metric]), value)
    assert endpoint_values == 120
    uc_pairs = list(csv.DictReader((DIRECTORIES[1] / 'all_paired_seed_endpoints.csv').open())); assert len(uc_pairs) == 180
    null_pairs = 0
    for row in uc_pairs:
        a = uc_points[(row['first'], int(row['seed']))]['metrics'][row['endpoint']]
        b = uc_points[(row['second'], int(row['seed']))]['metrics'][row['endpoint']]
        if a is None or b is None:
            assert a is None and b is None and row['difference_first_minus_second'] == ''; null_pairs += 1
        else: close(float(row['difference_first_minus_second']), a - b)
    assert null_pairs == 18
    for group in uc['model_summaries']:
        for metric, summary in group['metrics'].items():
            values = [uc_points[(group['system'], seed)]['metrics'][metric] for seed in [11, 22, 33]]
            assert values == summary['values']
            if any(v is None for v in values): assert all(v is None for v in values) and summary['mean'] is None
            else: close(summary['mean'], st.mean(values)); close(summary['sample_SD'], st.stdev(values))
    lo, port = 'dinov3_lora_ucf20', 'published_deeplab_fsl'
    pairs = {(x['first'], x['second'], x['endpoint']): x['summary']['values'] for x in uc['contrasts']}
    assert min(pairs[(lo, port, 'AUROC')]) > 0 and min(pairs[(lo, port, 'positive_image_Dice')]) > 0
    assert sum(x > 0 for x in pairs[(lo, port, 'pooled_native_pixel_AP')]) == 2
    assert sum(x < 0 for x in pairs[(lo, port, 'review_rate_at_source_threshold')]) == 2
    assert min(pairs[('dinov3_reference_ucf20', lo, 'mean_negative_field_pixel_FPR')]) > 0
    point22 = uc_points[(lo, 22)]; assert point22['negative_field_false_positive_pixels'] == 0
    close(point22['metrics']['false_positive_rate_at_source_threshold'], 81 / 86)
    uc_review = {'review_code': ref(Path(__file__).absolute()), 'result': ref(DIRECTORIES[1] / 'results.json'),
                 'checked_fits': 12, 'checked_per_fit_endpoints': 120, 'checked_paired_seed_endpoint_records': 180,
                 'checked_endpoint_summaries': 60, 'original_six_endpoint_summaries': 36,
                 'normal_structure_endpoint_values_retained_null': 12, 'normal_structure_paired_differences_retained_null': 18,
                 'findings': ['On the same144source-validation images/58positives/49provided families, LoRA AUROC0.908781 and Dice0.404889 exceed the full supervised port0.880914/0.307231 in all3paired fits.',
                              'LoRA native AP mean0.686674 exceeds port0.641029 but only2pairs are favorable and1is adverse. Its review fraction0.747685 is slightly higher than0.743056 despite the same source recall56/58;2seed pairs review less and1more.',
                              'LoRA seed22 has no predicted crack pixels on negative fields, yet its image branch flags81/86negative images at the selected source threshold. The18/58empty positive masks remain. Native negative-field specificity, image review and crack overlap are different endpoints.',
                              'Normal reference versus frozen changes mean Dice by+0.015974 with2positive/1adverse pair; AUROC changes by-0.020315 with1positive/2adverse. Reference versusLoRA has lower native AP and higher mean negative-field pixel FPR in all3pairs. No stable matching-specific gain is identified.',
                              'The86training and144validation images are contained within the574labeled UCF development/HPO pool, not230+574distinct images. Source validation selected checkpoints/calibration, so the observed improvements are retrospective, not independent confirmation.',
                              'All normal-structure FPR values remain not estimable. Negative-field FPR is separately reported in raw ratios and never imputed into the missing structure endpoint. Different objectives, pretraining and optimizer work remain confounded; raw fit timers are not isolated compute costs.'],
                 'raw_failed_fit_recovery': ref(BASE / 'operational_recovery_v1/selection_metadata_2026-10-27T00-31-35.243048+00-00/recovery.json'),
                 'new_model_fit_inference_or_bootstrap': False, 'independent_confirmation': False,
                 'figure_QA_and_native_article_integration_separate': True}
    immutable(DIRECTORIES[1] / 'numerical_review.json', uc_review)
    print(json.dumps({'ELPV_review': ref(DIRECTORIES[0] / 'numerical_review.json'),
                      'UCF_review': ref(DIRECTORIES[1] / 'numerical_review.json'), 'all_saved_values_checked': True}))


if __name__ == '__main__': main()
