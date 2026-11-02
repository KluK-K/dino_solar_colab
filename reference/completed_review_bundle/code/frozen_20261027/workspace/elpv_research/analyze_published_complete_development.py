"""Complete held-source cells, with unchanged source-selected published ports.

Consumes existing complete-cell registries and saved outputs only. No model,
inference, bootstrap, target calibration, new selection or final-cohort access.
"""
import argparse
import html
from pathlib import Path

import numpy as np

from analyze_dense_development_uncertainty import ENDPOINTS, saved_record
from analyze_published_complete_internal import close, summary, write_csv
from elresearch.common import ROOT, digest, now, read_json, save_json, sha256
from elresearch.confirmation_analysis import endpoint_value, pair_records
from elresearch.metrics import classification
from elresearch.publication_train import probability_metrics

BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'published_complete_development_readout_v1'
SEEDS = [11, 22, 33]
POPULATIONS = ['source_validation', 'held_acquisition_development']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-index', required=True)
    args = parser.parse_args()
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Development readouts must precede final confirmation')
    if not read_json(BASE / 'inference_efficiency_v1/status.json')['state'].startswith('queued'):
        raise PermissionError('Never compete with formal efficiency timings')
    audit_path = Path(args.audit_index)
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path
    if audit_path.name != 'completed_output_integrity_index_snapshot.json':
        raise ValueError('Bind an immutable saved-output audit snapshot')
    audits = read_json(audit_path)
    audited = {row['run']: row for row in audits['completed_published_fits_verified']}
    OUT.mkdir(exist_ok=True)
    protocol_path = BASE / 'published_methods_v1/protocol.json'
    dense_protocol_path = BASE / 'dense_development_v1/protocol.json'
    identity_path = BASE / 'source_development_identity_v1.json'
    dense_index_path = BASE / 'dense_development_complete_cells.json'
    protocol, dense_protocol, dense_index = map(read_json, [protocol_path, dense_protocol_path, dense_index_path])
    assert protocol['seeds'] == SEEDS
    assert dense_index['protocol_sha256'] == sha256(dense_protocol_path)
    assert protocol['development_design']['FSL_training_fractions'] == [.2, 1.]
    assert protocol['development_design']['SSL_training_fractions'] == [.2]
    identities = {row['id']: row for row in read_json(identity_path)['records']}
    selected_path = BASE / 'published_methods_v1/selected_hyperparameters.json'
    code_paths = [Path(__file__).absolute(), ROOT / 'analyze_dense_development_uncertainty.py',
                  ROOT / 'analyze_published_complete_internal.py', ROOT / 'elresearch/confirmation_analysis.py',
                  ROOT / 'elresearch/metrics.py', ROOT / 'elresearch/publication_train.py',
                  ROOT / 'build_dense_development_progress.py']
    def ref(path):
        return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
    plan = {'version': 1, 'kind': 'Retrospective complete held-source development cells with full published-method ports',
            'seeds': SEEDS, 'populations': POPULATIONS, 'endpoints': ENDPOINTS,
            'required_cells': [{'fold': cell['fold'], 'budget': cell['budget'],
                                'dense_cell_id': cell['cell_id'],
                                'dense_families': [m['family'] for m in cell['models']],
                                'published_views': ['published_deeplab_fsl/primary'] +
                                    (['published_psmt_ssl/primary', 'published_psmt_ssl/ensemble'] if cell['budget'] == .2 else [])}
                               for cell in dense_index['complete_cells'] if cell['fold'] != 'internal'],
            'bindings': [ref(p) for p in [*code_paths, protocol_path, dense_protocol_path,
                                          dense_index_path, identity_path, selected_path]],
            'completeness': 'Use the existing complete dense-cell registry. Require every applicable published model/view and all three seeds; never publish a partial cell.',
            'contrasts': 'Each originally declared dense family minus every applicable published view, separately in both populations. At20%, also SSL primary minus FSL, ensemble minus FSL, and ensemble minus SSL primary. Retain all six endpoints/three paired fitted seeds, including null endpoints.',
            'statistics': 'Every fit point, mean/sample SD and paired fitted-seed differences. SD is not CI. No bootstrap, p-value, adjusted significance, pixel pseudo-replication or independent population replication. Native pooled AP remains original whole-cohort point per fit.',
            'source_selection': 'Original checkpoints, view-specific source logit thresholds/temperatures and0.5 native cutoff remain fixed. No target-driven view/model/threshold/temperature/direction choice.',
            'interpretation': 'Both UCF and SCDD development inform study-wide design. These already observed acquisition cohorts are not untouched confirmation or verified physical modules. Different pretraining, objectives, score construction and optimizer work confound backbone-specific interpretations. Source/held differences also mix annotation/content/prevalence.',
            'exposure': 'Match actual training/validation identities within seed. State all training masks plus validation masks, SSL source-TRAIN unlabeled images and additional upstream UCF574 label/HPO exposure. No target in any fitting pool.',
            'timing': 'No runtime comparison; original fit timers are provenance only and may contain user pauses.',
            'replay_scope': 'Recompute saved image metrics/calibration and native integer consistency. Bind original source reload receipts. This does not rerun target model inference or recompute pooled AP from pixels.'}
    plan_path = OUT / 'plan.json'
    if plan_path.exists():
        assert read_json(plan_path) == plan, 'Bound analysis source/plan changed; preserve and explicitly amend'
    else:
        save_json(plan_path, plan)
    prior = read_json(OUT / 'index.json')['complete_cells'] if (OUT / 'index.json').exists() else []
    completed, pending = [], []
    for spec in plan['required_cells']:
        key = (spec['fold'], spec['budget'])
        old = next((x for x in prior if (x['fold'], x['budget']) == key), None)
        if old:
            assert sha256(ROOT / old['path']) == old['sha256']
            saved = read_json(ROOT / old['path'])
            for binding in saved['input_bindings']:
                assert sha256(ROOT / binding['path']) == binding['sha256']
            for binding in read_json(ROOT / old['outputs']['path'])['files']:
                assert sha256(ROOT / binding['path']) == binding['sha256']
            completed.append(old)
            continue
        cell = next(c for c in dense_index['complete_cells'] if (c['fold'], c['budget']) == key)
        split = next(s for s in dense_protocol['split_manifest']['folds'] if s['name'] == spec['fold'])
        published_split = next(s for s in protocol['split_manifest']['folds'] if s['name'] == spec['fold'])
        assert published_split == split, 'Use the same registered source/development split'
        ids = {'source_validation': split['source_validation_ids'],
               'held_acquisition_development': split['target_development_ids']}
        assert not set(ids[POPULATIONS[0]]) & set(ids[POPULATIONS[1]])
        assert all(identities[i]['dataset'] == 'scdd' for values in ids.values() for i in values)
        jobs = []
        for model in cell['models']:
            assert sorted(f['seed'] for f in model['fits']) == SEEDS
            for fit in model['fits']:
                run_path = ROOT / fit['run']
                assert sha256(run_path) == fit['run_sha256']
                assert sha256(ROOT / fit['predictions']) == fit['predictions_sha256']
                evaluation = run_path.parent / 'target_development.json'
                binding = next(x for x in cell['identity']['bindings'] if x['run'] == fit['run'])
                assert sha256(evaluation) == binding['evaluation_sha256']
                jobs.append({'system': model['family'], 'seed': fit['seed'], 'run': run_path,
                             'view': 'primary', 'published': False})
        missing = []
        for system in spec['published_views']:
            model, view = system.split('/')
            for seed in SEEDS:
                folder = BASE / 'published_methods_v1/runs' / spec['fold'] / f'budget_{spec["budget"]:g}' / f'{model}_seed{seed}'
                path = folder / 'run.json'
                if (not path.exists() or not (folder / 'target_development.json').exists()
                        or str(path.relative_to(ROOT)) not in audited):
                    missing.append(f'{system}/seed{seed}')
                else:
                    jobs.append({'system': system, 'seed': seed, 'run': path, 'view': view, 'published': True})
        if missing:
            pending.append({'fold': spec['fold'], 'budget': spec['budget'], 'missing_or_not_yet_audited': missing})
            continue
        bindings = {}
        def bind(path):
            value = ref(path); bindings[value['path']] = value
        for path in [plan_path, audit_path, ROOT / cell['cell_path']]:
            bind(path)
        assert sha256(ROOT / cell['cell_path']) == cell['cell_sha256']
        records = {p: {} for p in POPULATIONS}; rows = []; train_by_seed = {}; support = {}; truth = {}
        for job in jobs:
            path = job['run']; run = read_json(path); target_path = path.parent / 'target_development.json'
            target = read_json(target_path); published = job['published']; view = job['view']
            assert run['seed'] == job['seed'] and run['training_budget_fraction'] == spec['budget']
            assert run['checkpoint_sha256'] == target['checkpoint_sha256'] == sha256(path.parent / 'best.pt')
            assert not (path.parent / 'latest.pt').exists()
            assert run['study_id'] == (protocol if published else dense_protocol)['study_id']
            assert run['scientific_hashes'] == (protocol if published else dense_protocol)['scientific_hashes']
            for p in [path, path.parent / 'best.pt', target_path]:
                bind(p)
            if published:
                audit = audited[str(path.relative_to(ROOT))]; ap = ROOT / audit['verification']
                assert sha256(ap) == audit['sha256']; bind(ap)
                opening = path.parent / 'target_development_opening.json'; opened = read_json(opening)
                assert opened['run_sha256'] == sha256(path) and opened['target_ids'] == ids[POPULATIONS[1]]; bind(opening)
            training = sorted(run['training_ids']); unlabelled = run.get('unlabelled_source_ids', [])
            assert len(training) == run['training_images']
            assert set(training) | set(unlabelled) <= set(split['source_train_ids'])
            assert run['validation_ids'] == ids[POPULATIONS[0]]
            assert len(training) + len(ids[POPULATIONS[0]]) == run['total_pixel_annotated_images_including_validation']
            assert not set(training) & set(unlabelled)
            assert not (set(training) | set(unlabelled)) & (set(ids[POPULATIONS[0]]) | set(ids[POPULATIONS[1]]))
            if job['seed'] in train_by_seed:
                assert training == train_by_seed[job['seed']]
            else:
                train_by_seed[job['seed']] = training
            source = run['inference_views'][view] if published else run
            if view == 'primary':
                close(target['source_logit_threshold'], source['source_logit_threshold'])
            reload = source['checkpoint_reload']
            assert reload['fresh_model'] and max(reload['logit_max_abs_error'], reload['sampled_pixel_probability_max_abs_error']) <= 2e-5
            suffix = '_ensemble' if view == 'ensemble' else ''
            for population in POPULATIONS:
                is_source = population == POPULATIONS[0]
                pred = path.parent / (f'validation_predictions{suffix}.npz' if is_source else f'target_development_predictions{suffix}.npz')
                evaluated = source if is_source else (target['inference_views'][view] if published else target)
                bind(pred)
                with np.load(pred, allow_pickle=False) as z:
                    arrays = {k: z[k].copy() for k in ['ids', 'y', 'groups', 'logits']}
                record = saved_record({**run, 'source_logit_threshold': source['source_logit_threshold']}, evaluated,
                                      arrays, job['system'], ids[population], [identities[i]['split_group'] for i in ids[population]])
                local = evaluated['validation_localization'] if is_source else evaluated['localization']
                metrics = evaluated['validation_metrics'] if is_source else evaluated['metrics']
                calibration = evaluated['validation_calibration_temperature_scaled'] if is_source else evaluated['source_temperature_calibration']
                for k, value in classification(record['y'], record['logits'], record['source_threshold']).items():
                    close(value, metrics[k])
                for k, value in probability_metrics(record['y'], record['logits'], source['source_temperature']).items():
                    close(value, calibration[k])
                annotations = [(x['id'], x['known_pixels'], x['crack_pixels'], x['negative_image_normal_structure_pixels']) for x in local['per_image']]
                if population in support:
                    assert annotations == support[population] and np.array_equal(record['y'], truth[population])
                else:
                    support[population] = annotations; truth[population] = record['y']
                for x in local['per_image']:
                    tp = x['predicted_crack_pixels'] - x['false_positive_pixels']; union = x['predicted_crack_pixels'] + x['crack_pixels'] - tp
                    close(tp / union if union else 1., x['iou'])
                records[population].setdefault(job['system'], []).append(record)
                positive = [x for x in local['per_image'] if x['crack_positive']]
                rows.append({'fold': spec['fold'], 'budget': spec['budget'], 'population': population,
                             'system': job['system'], 'seed': job['seed'], 'run': str(path.relative_to(ROOT)),
                             'checkpoint_sha256': run['checkpoint_sha256'], 'training_masks': len(training),
                             'validation_masks': len(ids[POPULATIONS[0]]), 'unlabelled_source_images': len(unlabelled),
                             'source_threshold': record['source_threshold'], 'source_temperature': source['source_temperature'],
                             'images': len(record['ids']), 'positive_images': int(record['y'].sum()),
                             'provided_capture_families': len(set(record['groups'].tolist())),
                             'empty_positive_images': sum(x['predicted_crack_pixels'] == 0 for x in positive),
                             'zero_overlap_positive_images': sum(x['predicted_crack_pixels'] == x['false_positive_pixels'] for x in positive),
                             **{e: endpoint_value(record, e) for e in ENDPOINTS}})
        contrasts = []; pairs = [(d, p) for d in spec['dense_families'] for p in spec['published_views']]
        if spec['budget'] == .2:
            pairs += [('published_psmt_ssl/primary', 'published_deeplab_fsl/primary'),
                      ('published_psmt_ssl/ensemble', 'published_deeplab_fsl/primary'),
                      ('published_psmt_ssl/ensemble', 'published_psmt_ssl/primary')]
        paired_rows = []; systems = {}
        for population in POPULATIONS:
            systems[population] = {}
            for system, values in records[population].items():
                values.sort(key=lambda x: x['model']['seed']); assert [x['model']['seed'] for x in values] == SEEDS
                systems[population][system] = {e: summary([endpoint_value(v, e) for v in values]) for e in ENDPOINTS}
            for first, second in pairs:
                paired = pair_records(records[population][first], records[population][second]); endpoints = {}
                for endpoint in ENDPOINTS:
                    a = [endpoint_value(x, endpoint) for x, _ in paired]; z = [endpoint_value(y, endpoint) for _, y in paired]
                    delta = [x-y if x is not None and y is not None else None for x, y in zip(a, z)]
                    endpoints[endpoint] = {'first_values': a, 'second_values': z, 'first_minus_second': summary(delta)}
                    for seed, x, y, difference in zip(SEEDS, a, z, delta):
                        paired_rows.append({'fold': spec['fold'], 'budget': spec['budget'], 'population': population,
                                            'first': first, 'second': second, 'endpoint': endpoint, 'seed': seed,
                                            'first_value': x, 'second_value': y, 'difference_raw_ratio': difference})
                contrasts.append({'population': population, 'first': first, 'second': second, 'endpoints': endpoints})
        identity = {'plan_sha256': sha256(plan_path), 'cell': spec, 'input_bindings': sorted(bindings.values(), key=lambda x: x['path'])}
        directory = OUT / digest(identity)[:20]; directory.mkdir(exist_ok=True)
        result_path = directory / 'results.json'; assert not result_path.exists()
        result = {'at': now(), 'identity': identity, 'plan': ref(plan_path), 'fold': spec['fold'], 'budget': spec['budget'],
                  'input_bindings': identity['input_bindings'], 'unique_fits': len({x['run'] for x in rows}),
                  'correlated_fit_view_population_records': len(rows), 'systems': systems, 'all_seed_records': rows,
                  'paired_model_population_contrasts': len(contrasts), 'endpoint_summaries': len(contrasts)*len(ENDPOINTS),
                  'comparisons': contrasts, 'independent_confirmation': False, 'new_model_inference_or_fit_or_bootstrap': False}
        save_json(result_path, result)
        write_csv(directory / 'all_seed_population_outcomes.csv', rows)
        write_csv(directory / 'all_paired_differences.csv', paired_rows)
        content = ['<h1>Complete observed development cell</h1>', '<p>'+html.escape(spec['fold'])+f' / {spec["budget"]:g} source-training fraction; all applicable systems/seeds. No independent confirmation, model inference or confidence interval.</p>']
        for population, values in systems.items():
            content += ['<h2>'+population+'</h2><table><tr><th>System</th>'+''.join('<th>'+e+'</th>' for e in ENDPOINTS)+'</tr>']
            for system, endpoints in values.items():
                content.append('<tr><td>'+html.escape(system)+'</td>'+''.join('<td>'+('NE' if endpoints[e]['mean'] is None else f'{endpoints[e]["mean"]:.6g}')+'</td>' for e in ENDPOINTS)+'</tr>')
            content.append('</table>')
        content += ['<p>All values are ratios; source and held populations differ. AP is point-only. Source thresholds and temperatures are unchanged. Empty/zero-overlap positives, every seed and all paired differences are retained in CSV/JSON. Original upstream UCF exposure and objective/scoring/pretraining confounds remain.</p>']
        (directory / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete development readout</title><style>body{font:15px Arial;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>'+''.join(content)+'</html>')
        outputs = directory / 'outputs.json';save_json(outputs, {'at':now(),'files':[ref(directory/name) for name in ['results.json','all_seed_population_outcomes.csv','all_paired_differences.csv','report.html']]})
        completed.append({**{k:spec[k] for k in ['fold','budget']},**ref(result_path),'outputs':ref(outputs)})
    save_json(OUT / 'index.json', {'at':now(),'plan':ref(plan_path),'complete_cells':completed,'planned_cells':len(plan['required_cells']),'pending_cells':pending,'new_inference_or_fit':False,'final_confirmation':False})
    print({'complete_cells':len(completed),'planned_cells':len(plan['required_cells']),'pending_cells':len(pending)})


if __name__ == '__main__':
    main()
