"""Complete twelve-fit matched UCF source readout; saved outputs only."""
from pathlib import Path
import csv
import hashlib
import html
import json

import numpy as np
from analyze_dense_development_uncertainty import saved_record, ENDPOINTS
from analyze_published_complete_internal import close, summary
from elresearch.confirmation_analysis import endpoint_value, pair_records
from elresearch.metrics import classification
from elresearch.publication_train import probability_metrics

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'published_methods_v1'
OUT = BASE / 'matched_UCF_source_readout_v1'
DINO_AUDIT = BASE / 'operational_wait_observations/matched_DINO_actual_audit_2026-10-27T05-02-00.848111+00-00/immutable_index_snapshot.json'
PORT_AUDIT = BASE / 'operational_wait_observations/published60_audit_2026-10-27T00-25-36.207470+00-00/completed_output_integrity_index_snapshot.json'
SYSTEMS = ['published_deeplab_fsl', 'dinov3_frozen_ucf20', 'dinov3_lora_ucf20', 'dinov3_reference_ucf20']
EXTRA = ['average_precision', 'false_positive_rate_at_source_threshold', 'positive_image_IoU', 'mean_negative_field_pixel_FPR']
PAIRS = [(x, SYSTEMS[0]) for x in SYSTEMS[1:]] + [(SYSTEMS[2], SYSTEMS[1]), (SYSTEMS[3], SYSTEMS[1]), (SYSTEMS[3], SYSTEMS[2])]


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def immutable(p, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists(): assert p.read_text() == text
    else: p.write_text(text)


def write_csv(p, rows):
    with p.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    assert not (BASE / 'final_confirmation_v1/observation_opened.json').exists()
    assert read(BASE / 'inference_efficiency_v1/status.json')['state'] == 'complete'
    dino_audits = read(DINO_AUDIT); assert dino_audits['all_nine_saved_output_audits_complete']
    dino_receipts = {x['run']: x['verification'] for x in dino_audits['receipts']}
    port_receipts = {x['run']: {'path': x['verification'], 'sha256': x['sha256']} for x in read(PORT_AUDIT)['completed_published_fits_verified']}
    identity = {x['id']: x for x in read(BASE / 'source_development_identity_v1.json')['records']}
    paths = [Path(__file__).absolute(), ROOT / 'analyze_dense_development_uncertainty.py',
             ROOT / 'analyze_published_complete_internal.py', ROOT / 'elresearch/confirmation_analysis.py',
             ROOT / 'elresearch/metrics.py', ROOT / 'elresearch/publication_train.py', DINO_AUDIT, PORT_AUDIT,
             SOURCE / 'protocol.json', SOURCE / 'matched_dino_controls.json', SOURCE / 'selected_hyperparameters.json',
             BASE / 'source_development_identity_v1.json', BASE / 'dense_development_v1/selected_hyperparameters.json']
    jobs = []
    for system in SYSTEMS:
        for seed in [11, 22, 33]:
            folder = SOURCE / 'source_transfer' / f'{system}_seed{seed}'; path = folder / 'run.json'
            receipt = (port_receipts if system == SYSTEMS[0] else dino_receipts)[str(path.relative_to(ROOT))]
            assert sha(ROOT / receipt['path']) == receipt['sha256']
            paths += [ROOT / receipt['path'], *[folder / name for name in ['run.json', 'validation_predictions.npz', 'best.pt', 'history.json']]]
            if (folder / 'normal_bank_recipe.json').exists(): paths.append(folder / 'normal_bank_recipe.json')
            jobs.append((system, seed, folder))
    endpoints = [*ENDPOINTS, *EXTRA]
    plan = {'version': 1, 'kind': 'Complete matched UCF source-validation comparison',
            'bindings': [ref(p) for p in sorted(set(paths))], 'systems': SYSTEMS, 'seeds': [11, 22, 33],
            'endpoints': endpoints, 'original_six_endpoints': ENDPOINTS, 'contrasts': PAIRS,
            'exposure': '86 UCF source-training masks +144 source-validation masks per fit, with matched training IDs within each fitted seed. Upstream source HPO has used all574 UCF labels;230 is not the total study annotation exposure.',
            'statistics': 'All3per-fit points, mean/sample SD and all paired differences retained. No bootstrap, p-values, CIs, adjusted significance or independent domain replication. Pooled native AP is the original whole-cohort point, never inferred by bootstrapping pixels or averaging per-image AP.',
            'null_policy': 'UCF supplies no normal-structure subsets: all normal_structure_pixel_FPR values and contrasts must remain null/not estimable. Mean negative-field pixel FPR is separately named and is not a substitute for the missing structure-specific endpoint.',
            'thresholds': 'Every original source logit threshold and temperature preserved; native cutoff0.5. No source outcome-driven refit or retuning and no target scoring.',
            'failure_policy': 'All144images and58positive cases retained per fit. Report positive-empty, positive-zero-overlap, all-image-empty, all negative-field false-positive pixels and endpoint distinctions.',
            'matching_limit': 'Same masks/validation IDs do not equate pretraining, loss, architecture, optimization, decoder training or all upstream selection. No matching-specific causal mechanism or architecture-only superiority.',
            'timing_limit': 'Raw fit times are provenance, not isolated active compute. The first frozen seed11fit was finalized by the registered metadata-only recovery with19rounds/950updates preserved; failed finalization work is not fully represented in its restored fit timer.',
            'scope': 'The same144observed UCF validation images selected checkpoints and source calibration and have influenced method design. This is retrospective source evidence, not independent confirmation.'}
    OUT.mkdir(exist_ok=True); immutable(OUT / 'plan.json', plan)
    destination = OUT / digest(plan)[:20]
    if destination.exists():
        for b in read(destination / 'outputs.json')['files']: assert sha(ROOT / b['path']) == b['sha256']
        print(json.dumps({'reused': True, 'result': ref(destination / 'results.json')})); return
    destination.mkdir()
    points, records, reference_train, reference_annotation = [], {}, {}, None
    truth = None; ids = None; group_ids = None
    for system, seed, folder in jobs:
        run = read(folder / 'run.json'); assert run['seed'] == seed and run['training_budget_fraction'] == .2
        assert run['training_images'] == len(run['training_ids']) == 86
        assert run['validation_images'] == len(run['validation_ids']) == 144
        assert run['total_pixel_annotated_images_including_validation'] == 230
        assert sha(folder / 'best.pt') == run['checkpoint_sha256']
        training = sorted(run['training_ids'])
        if seed in reference_train: assert training == reference_train[seed]
        else: reference_train[seed] = training
        if ids is None:
            ids = run['validation_ids']; group_ids = [identity[i]['split_group'] for i in ids]
        assert run['validation_ids'] == ids and not set(training) & set(ids)
        evaluated = run['inference_views']['primary'] if system == SYSTEMS[0] else run
        view = {**run, 'source_logit_threshold': evaluated['source_logit_threshold']}
        with np.load(folder / 'validation_predictions.npz', allow_pickle=False) as values:
            arrays = {k: values[k].copy() for k in ['ids', 'y', 'groups', 'logits']}
        record = saved_record(view, evaluated, arrays, system, ids, group_ids)
        if truth is None: truth = arrays['y']
        np.testing.assert_array_equal(truth, arrays['y']); assert int(truth.sum()) == 58
        local = evaluated['validation_localization']; native = local['per_image']
        annotation = [(x['id'], x['known_pixels'], x['crack_pixels'], x.get('negative_image_normal_structure_pixels')) for x in native]
        if reference_annotation is None: reference_annotation = annotation
        assert annotation == reference_annotation and all(x[3] is None for x in annotation)
        for key, value in classification(record['y'], record['logits'], record['source_threshold']).items(): close(value, evaluated['validation_metrics'][key])
        for key, value in probability_metrics(record['y'], record['logits'], evaluated['source_temperature']).items(): close(value, evaluated['validation_calibration_temperature_scaled'][key])
        values = {name: endpoint_value(record, name) for name in endpoints if name != EXTRA[-1]}
        values[EXTRA[-1]] = local['mean_negative_image_false_positive_fraction']
        assert values['normal_structure_pixel_FPR'] is None
        positives = [x for x in native if x['crack_positive']]; negatives = [x for x in native if not x['crack_positive']]
        assert len(positives) == 58 and len(negatives) == 86
        close(values[EXTRA[-1]], float(np.mean([x['false_positive_pixels'] / x['known_pixels'] for x in negatives])))
        point = {'system': system, 'seed': seed, 'training_masks': 86, 'validation_masks': 144,
                 'training_ID_set_sha256': digest(training), 'checkpoint_sha256': run['checkpoint_sha256'],
                 'source_threshold': record['source_threshold'], 'source_temperature': evaluated['source_temperature'],
                 'trainable_parameters': run['trainable_parameters'], 'completed_rounds': run['completed_rounds'],
                 'optimizer_updates': run['optimizer_updates'], 'raw_fit_seconds': run['training_and_validation_seconds_excluding_preparation'],
                 'timer_is_isolated_active_compute': False, 'metrics': values,
                 'positive_empty_predictions': sum(x['predicted_crack_pixels'] == 0 for x in positives),
                 'positive_zero_overlap': sum(x['predicted_crack_pixels'] - x['false_positive_pixels'] == 0 for x in positives),
                 'all_image_empty_predictions': sum(x['predicted_crack_pixels'] == 0 for x in native),
                 'negative_field_false_positive_pixels': sum(x['false_positive_pixels'] for x in negatives),
                 'source_run': ref(folder / 'run.json')}
        points.append(point); records.setdefault(system, []).append(record)
    groups, comparisons, paired_rows = [], [], []
    for system in SYSTEMS:
        rows = [p for p in points if p['system'] == system]
        assert [x['seed'] for x in rows] == [11, 22, 33]
        groups.append({'system': system, 'metrics': {name: summary([p['metrics'][name] for p in rows]) for name in endpoints}})
    for first, second in PAIRS:
        pairs = pair_records(records[first], records[second]); assert len(pairs) == 3
        for name in endpoints:
            a = [p for p in points if p['system'] == first]; b = [p for p in points if p['system'] == second]
            differences = []
            for left, right in zip(a, b):
                assert left['seed'] == right['seed']
                x, y = left['metrics'][name], right['metrics'][name]
                difference = x - y if x is not None and y is not None else None
                differences.append(difference); paired_rows.append({'first': first, 'second': second, 'seed': left['seed'], 'endpoint': name, 'first_value': x, 'second_value': y, 'difference_first_minus_second': difference})
            comparisons.append({'first': first, 'second': second, 'endpoint': name, 'paired_seeds': [11, 22, 33], 'summary': summary(differences), 'interval': None})
    value = {'plan': ref(OUT / 'plan.json'), 'source_validation_images': 144, 'positive_images': 58,
             'provided_validation_families': len(set(group_ids)), 'per_fit': points, 'model_summaries': groups,
             'contrasts': comparisons, 'counts': {'fits': 12, 'systems': 4, 'paired_contrasts': len(PAIRS),
                 'original_six_endpoint_summaries': len(PAIRS) * len(ENDPOINTS), 'all_endpoint_summaries': len(comparisons),
                 'per_fit_endpoint_values': len(points) * len(endpoints), 'paired_seed_endpoint_records': len(paired_rows)},
             'independent_confirmation': False, 'new_model_fit_inference_or_bootstrap': False}
    immutable(destination / 'results.json', value)
    write_csv(destination / 'all_fit_endpoints.csv', [{'system': p['system'], 'seed': p['seed'], **p['metrics']} for p in points])
    write_csv(destination / 'all_paired_seed_endpoints.csv', paired_rows)
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete matched UCF source readout</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}</style><h1>All12matched UCF source fits</h1>']
    for key in ['exposure', 'scope', 'null_policy', 'statistics', 'matching_limit', 'timing_limit']: page.append('<p>' + html.escape(plan[key]) + '</p>')
    for name in endpoints:
        page.append('<h2>' + name + '</h2><table><tr><th>System</th><th>Seed11</th><th>Seed22</th><th>Seed33</th><th>Mean</th><th>Sample SD (not CI)</th></tr>')
        for group in groups:
            d = group['metrics'][name]; fmt = lambda x: 'not estimable' if x is None else f'{x:.6f}'
            page.append('<tr><td>' + group['system'] + '</td>' + ''.join('<td>' + fmt(x) + '</td>' for x in [*d['values'], d['mean'], d['sample_SD']]) + '</tr>')
        page.append('</table>')
    page.append('<h2>Native failure counts</h2><table><tr><th>System/seed</th><th>Empty positive masks /58</th><th>Zero overlap /58</th><th>All-image empty /144</th><th>Negative-field FP pixels</th></tr>')
    for p in points: page.append(f"<tr><td>{p['system']} / {p['seed']}</td><td>{p['positive_empty_predictions']}</td><td>{p['positive_zero_overlap']}</td><td>{p['all_image_empty_predictions']}</td><td>{p['negative_field_false_positive_pixels']}</td></tr>")
    page.append('</table><p>Every declared contrast, endpoint and fitted seed remains in machine-readable outputs; unsupported normal-structure endpoints stay null. No final confirmation score was accessed.</p></html>')
    (destination / 'report.html').write_text('\n'.join(page))
    immutable(destination / 'outputs.json', {'files': [ref(p) for p in sorted(destination.iterdir()) if p.is_file()]})
    immutable(OUT / 'latest.json', {'plan': ref(OUT / 'plan.json'), 'result': ref(destination / 'results.json'), 'outputs': ref(destination / 'outputs.json'), 'report': ref(destination / 'report.html'), 'counts': value['counts']})
    print(json.dumps({'result': ref(destination / 'results.json'), 'counts': value['counts']}))


if __name__ == '__main__': main()
