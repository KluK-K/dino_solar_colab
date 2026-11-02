"""All original 24 nested-development ELPV fits, without new fitting/inference."""
from pathlib import Path
import csv
import hashlib
import html
import json
import os
import statistics as st

import numpy as np
from elresearch.metrics import classification

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'elpv_review_supplement_v1'
OUT = BASE / 'complete_ELPV_supplement_readout_v1'
AUDIT = BASE / 'operational_wait_observations/ELPV24_completed_output_audit_2026-10-27T05-13-39.137211+00-00/summary.json'
METRICS = ['auc', 'average_precision', 'macro_f1', 'balanced_accuracy', 'accuracy', 'recall', 'precision', 'review_rate']
FEATURES = ['hog_stats', 'dinov3_cls']
CLASSIFIERS = ['logistic', 'linear_svm', 'rbf_svm']


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return {'path': os.path.relpath(p, ROOT), 'sha256': sha(p)}
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def immutable(p, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists(): assert p.read_text() == text
    else: p.write_text(text)


def summary(values):
    return {'values': values, 'mean': st.mean(values), 'sample_SD': st.stdev(values),
            'minimum': min(values), 'maximum': max(values), 'confidence_interval': None,
            'interpretation': 'Four overlapping-training outer CV folds, not independent datasets or seed replications.'}


def csv_file(p, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with p.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, keys); writer.writeheader(); writer.writerows(rows)


def main():
    assert not (BASE / 'final_confirmation_v1/observation_opened.json').exists()
    assert sha(AUDIT) == '02d40b7c5983747a5fb3c4ca1559dd0877f7d3a3fbd81cbc2ba2880f319c3d1e'
    audit = read(AUDIT); assert audit['verified_outer_fits'] == 24
    for key in ['source_bindings', 'result', 'original_verification', 'checker']:
        binding = audit[key]; assert sha(ROOT / binding['path']) == binding['sha256']
    results, protocol = read(SOURCE / 'results.json'), read(SOURCE / 'protocol.json')
    rows = read(ROOT / 'data/elpv/manifest.json')['records']
    assert len(rows) == 1968 and len({r['id'] for r in rows}) == 1968
    identities = {r['id']: r for r in rows}
    inputs = [Path(__file__).absolute(), AUDIT, SOURCE / 'protocol.json', SOURCE / 'results.json',
              SOURCE / 'verification.json', SOURCE / 'embedding_identity.json', SOURCE / 'dinov3_cls.npz',
              ROOT / 'data/elpv/manifest.json', ROOT / 'elresearch/metrics.py', ROOT / 'elresearch/elpv_supplement.py']
    for binding in protocol['source_bindings']:
        p = Path(binding['path']); assert sha(p) == binding['sha256']; inputs.append(p)
    fits = results['fits']; assert len(fits) == 24
    for run in fits:
        folder = SOURCE / 'fits' / f"{run['feature']}_{run['classifier']}_outer{run['outer_fold']}"
        inputs += [folder / name for name in ['run.json', 'model.joblib', 'predictions.npz']]
    plan = {'version': 1, 'kind': 'Complete ELPV review-label nested-development supplement',
            'bindings': [ref(p) for p in sorted(set(inputs))], 'features': FEATURES,
            'classifiers': CLASSIFIERS, 'outer_folds': [0, 1, 2, 3], 'metrics': METRICS,
            'primary': 'Original per-outer-fold ROC-AUC and every fold value; unweighted descriptive mean/sample SD only.',
            'paired_contrasts': 'DINOv3 CLS minus HOG/statistics, separately for each originally registered classifier and all8original saved endpoints, retaining all4same-fold differences.',
            'threshold': 'Original classifier decision_function threshold0 for thresholded endpoints. No sensitivity matching, label-driven recalibration, threshold changes or class-direction flips.',
            'labels': 'Original expert_label>0 requires review, not crack presence, severity, box truth or pixel masks. ELPV1968 previously studied development only; all656course test images remain unscored.',
            'inner_fit_scope': audit['inner_count_scope'],
            'statistics': 'Folds share training images; their SD and paired differences are descriptive, not independence-based uncertainty, p-values, significance or independent population replication. No pooled OOF AUROC/AP: raw margins from independently selected outer fits are not assumed cross-fold calibrated.',
            'bias_and_scope': 'Feature dimension1806HOG/statistics versus384pretrainedCLS, pretraining, feature construction and earlier exposure differ. This matched-classifier supplement is not a CNN comparison or a causal architecture-only test. Older course scores used a different selection protocol and must not be ranked as equivalent.',
            'replay': 'Recompute all saved outer metrics from existing predictions; inspect inner/outer identity and group separation. No new model inference/reload, fitting, bootstrap or final-confirmation scoring.'}
    OUT.mkdir(exist_ok=True); immutable(OUT / 'plan.json', plan)
    destination = OUT / digest(plan)[:20]
    if destination.exists():
        for binding in read(destination / 'outputs.json')['files']: assert sha(ROOT / binding['path']) == binding['sha256']
        print(json.dumps({'reused': True, 'result': ref(destination / 'results.json')})); return
    destination.mkdir()
    per_fit, per_metric, groups, differences = [], [], [], []
    model_rows = {}
    for feature in FEATURES:
        for classifier in CLASSIFIERS:
            selected = sorted([x for x in fits if (x['feature'], x['classifier']) == (feature, classifier)], key=lambda x: x['outer_fold'])
            assert [x['outer_fold'] for x in selected] == [0, 1, 2, 3]
            validation_coverage = []
            for run in selected:
                folder = SOURCE / 'fits' / f"{feature}_{classifier}_outer{run['outer_fold']}"
                assert read(folder / 'run.json') == run
                assert sha(folder / 'predictions.npz') == run['predictions_sha256']
                assert sha(folder / 'model.joblib') == run['checkpoint_sha256']
                training_ids, held = run['training_ids'], run['held_out_development_ids']
                assert not set(training_ids) & set(held) and set(training_ids) | set(held) == set(identities)
                assert not {identities[i]['split_group'] for i in training_ids} & {identities[i]['split_group'] for i in held}
                coverage = np.zeros(len(training_ids), dtype=int)
                for inner in run['inner_folds']:
                    a, b = inner['train'], inner['validation']
                    assert not set(a) & set(b) and set(a) | set(b) == set(range(len(training_ids)))
                    assert not {identities[training_ids[i]]['split_group'] for i in a} & {identities[training_ids[i]]['split_group'] for i in b}
                    coverage[b] += 1
                assert len(run['inner_folds']) == 3 and np.all(coverage == 1)
                with np.load(folder / 'predictions.npz', allow_pickle=False) as values:
                    assert values['ids'].tolist() == held
                    np.testing.assert_array_equal(values['y'], [identities[i]['target'] for i in held])
                    metrics = classification(values['y'], values['scores'], 0.)
                    assert metrics == run['outer_metrics_at_default_zero']
                validation_coverage.extend(held)
                point = {'feature': feature, 'classifier': classifier, 'outer_fold': run['outer_fold'],
                         'training_images': len(training_ids), 'held_images': len(held),
                         'held_positive': metrics['positive'], 'held_negative': metrics['n'] - metrics['positive'],
                         'provided_training_groups': len({identities[i]['split_group'] for i in training_ids}),
                         'provided_held_groups': len({identities[i]['split_group'] for i in held}),
                         'best_params': run['best_params'], 'inner_selected_AUROC': run['inner_selected_AUROC'],
                         'metrics_at_default_zero': metrics, 'source_run': ref(folder / 'run.json')}
                per_fit.append(point)
                for metric in METRICS: per_metric.append({'feature': feature, 'classifier': classifier, 'outer_fold': run['outer_fold'], 'metric': metric, 'value': metrics[metric]})
            assert len(validation_coverage) == len(set(validation_coverage)) == 1968
            model_rows[(feature, classifier)] = selected
            groups.append({'feature': feature, 'classifier': classifier, 'outer_folds': [0, 1, 2, 3],
                           'metrics': {metric: summary([x['outer_metrics_at_default_zero'][metric] for x in selected]) for metric in METRICS}})
    for classifier in CLASSIFIERS:
        first, second = model_rows[('dinov3_cls', classifier)], model_rows[('hog_stats', classifier)]
        for a, b in zip(first, second):
            assert a['training_ids'] == b['training_ids'] and a['held_out_development_ids'] == b['held_out_development_ids']
            assert a['inner_folds'] == b['inner_folds']
        for metric in METRICS:
            values = [a['outer_metrics_at_default_zero'][metric] - b['outer_metrics_at_default_zero'][metric] for a, b in zip(first, second)]
            differences.append({'classifier': classifier, 'first': 'dinov3_cls', 'second': 'hog_stats', 'metric': metric,
                                'outer_folds': [0, 1, 2, 3], 'summary': summary(values)})
    value = {'plan': ref(OUT / 'plan.json'), 'per_fit': per_fit, 'model_summaries': groups,
             'paired_feature_contrasts': differences, 'counts': {'outer_fits': 24, 'registered_inner_grid_fits': 480,
                 'unique_reused_development_images': 1968, 'model_systems': 6, 'original_saved_endpoints': 8,
                 'per_fit_metric_values': len(per_metric), 'same_classifier_feature_endpoint_summaries': len(differences),
                 'same_fold_feature_metric_differences': sum(len(d['summary']['values']) for d in differences)},
             'course_test_scored': False, 'independent_confirmation': False, 'new_model_fit_inference_or_bootstrap': False}
    immutable(destination / 'results.json', value)
    csv_file(destination / 'all_outer_fold_metrics.csv', per_metric)
    csv_file(destination / 'all_paired_feature_differences.csv', [{'classifier': x['classifier'], 'metric': x['metric'], 'outer_fold': fold, 'difference': difference} for x in differences for fold, difference in zip(x['outer_folds'], x['summary']['values'])])
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete ELPV nested-development supplement</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}</style><h1>Complete ELPV review-label supplement</h1>']
    for key in ['labels', 'threshold', 'statistics', 'bias_and_scope', 'inner_fit_scope']: page.append('<p>' + html.escape(plan[key]) + '</p>')
    for metric in METRICS:
        page.append('<h2>' + metric + '</h2><table><tr><th>System</th><th>Fold0</th><th>Fold1</th><th>Fold2</th><th>Fold3</th><th>Mean</th><th>Sample SD (not CI)</th></tr>')
        for group in groups:
            d = group['metrics'][metric]
            page.append('<tr><td>' + group['feature'] + ' / ' + group['classifier'] + '</td>' + ''.join(f'<td>{x:.6f}</td>' for x in [*d['values'], d['mean'], d['sample_SD']]) + '</tr>')
        page.append('</table><h3>DINO CLS minus HOG/statistics, same classifier/fold</h3><table><tr><th>Classifier</th><th>Fold0</th><th>Fold1</th><th>Fold2</th><th>Fold3</th><th>Mean difference</th></tr>')
        for contrast in differences:
            if contrast['metric'] != metric: continue
            d = contrast['summary']; page.append('<tr><td>' + contrast['classifier'] + '</td>' + ''.join(f'<td>{x:+.6f}</td>' for x in [*d['values'], d['mean']]) + '</tr>')
        page.append('</table>')
    page.append('<p>All model choices and source outputs remain unchanged. No pooled cross-fold ranking metric, p-value, bootstrap interval or independent confirmation is asserted. Native crack localization, final figures and manuscript integration remain separate.</p></html>')
    (destination / 'report.html').write_text('\n'.join(page))
    immutable(destination / 'outputs.json', {'files': [ref(p) for p in sorted(destination.iterdir()) if p.is_file()]})
    immutable(OUT / 'latest.json', {'plan': ref(OUT / 'plan.json'), 'result': ref(destination / 'results.json'), 'outputs': ref(destination / 'outputs.json'), 'report': ref(destination / 'report.html'), 'counts': value['counts']})
    print(json.dumps({'result': ref(destination / 'results.json'), 'counts': value['counts']}))


if __name__ == '__main__': main()
