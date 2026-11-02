"""All-model/three-seed internal source readout; no new inference or bootstrap."""
import csv
import html
import json
from pathlib import Path
import statistics

import numpy as np

from analyze_dense_development_uncertainty import saved_record, ENDPOINTS
from elresearch.common import ROOT, now, read_json, save_json, sha256, digest
from elresearch.confirmation_analysis import endpoint_value, pair_records
from elresearch.metrics import classification
from elresearch.publication_train import probability_metrics

BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'published_complete_internal_readout_v1'
SEEDS = [11, 22, 33]
DENSE_FAMILIES = ['deeplabv3plus_full', 'dinov3_dense_frozen', 'dinov3_dense_lora',
                  'dinov3_dense_full', 'dinov3_dense_reference',
                  'dinov3_reference_no_position', 'dinov3_reference_mean_evidence']
AUDITS = BASE / 'operational_wait_observations/published30_audit_2026-10-21T09-29-42.246249+00-00/completed_output_integrity_index_snapshot.json'


def close(first, second):
    if first is None or second is None:
        if first is not second:
            raise ValueError('Null endpoint changed')
    elif not np.isclose(first, second, rtol=0, atol=2e-12):
        raise ValueError(f'Saved endpoint differs: {first}/{second}')


def summary(values):
    return {'values': values, 'available': sum(x is not None for x in values),
            'mean': statistics.mean(values) if all(x is not None for x in values) else None,
            'sample_SD': statistics.stdev(values) if all(x is not None for x in values) else None,
            'SD_is_confidence_interval': False}


def write_csv(path, rows):
    with path.open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Retrospective source readout must precede final scoring')
    efficiency = read_json(BASE / 'inference_efficiency_v1/status.json')
    if not (efficiency['state'].startswith('queued') or efficiency['state'] == 'complete'):
        raise PermissionError('Do not compete with formal efficiency measurements')
    OUT.mkdir(exist_ok=True)
    bindings = {}

    def bind(path):
        path = Path(path)
        value = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
        bindings[value['path']] = value
        return value

    index_path = BASE / 'dense_development_complete_cells.json'
    dense_protocol_path = BASE / 'dense_development_v1/protocol.json'
    published_protocol_path = BASE / 'published_methods_v1/protocol.json'
    identity_path = BASE / 'source_development_identity_v1.json'
    for path in [index_path, dense_protocol_path, published_protocol_path, identity_path, AUDITS,
                 ROOT / 'analyze_dense_development_uncertainty.py',
                 ROOT / 'elresearch/confirmation_analysis.py', ROOT / 'elresearch/metrics.py',
                 ROOT / 'elresearch/publication_train.py', Path(__file__)]:
        bind(path)
    dense_index = read_json(index_path)
    dense_protocol = read_json(dense_protocol_path)
    published_protocol = read_json(published_protocol_path)
    assert dense_index['protocol_sha256'] == sha256(dense_protocol_path)
    assert published_protocol['seeds'] == SEEDS
    split = next(x for x in dense_protocol['split_manifest']['folds'] if x['name'] == 'internal')
    ids = split['source_validation_ids']
    source_ids = {x['id']: x for x in read_json(identity_path)['records']}
    assert all(i in source_ids and source_ids[i]['dataset'] == 'scdd' for i in ids)
    groups = [source_ids[i]['split_group'] for i in ids]
    audited = {x['run']: x for x in read_json(AUDITS)['completed_published_fits_verified']}
    cells = []
    for budget in [.2, 1.]:
        cell = next(x for x in dense_index['complete_cells'] if x['fold'] == 'internal' and x['budget'] == budget)
        assert [x['family'] for x in cell['models']] == DENSE_FAMILIES
        path = ROOT / cell['cell_path']
        assert sha256(path) == cell['cell_sha256']
        bind(path)
        jobs = []
        for model in cell['models']:
            assert sorted(x['seed'] for x in model['fits']) == SEEDS
            for fit in model['fits']:
                run_path = ROOT / fit['run']
                prediction = ROOT / fit['predictions']
                assert sha256(run_path) == fit['run_sha256'] and sha256(prediction) == fit['predictions_sha256']
                jobs.append({'system': model['family'], 'seed': fit['seed'], 'run': run_path,
                             'predictions': prediction, 'view': 'primary', 'published': False})
        published = [('published_deeplab_fsl', ['primary'])]
        if budget == .2:
            published.append(('published_psmt_ssl', ['primary', 'ensemble']))
        for model, views in published:
            for seed in SEEDS:
                run_path = BASE / 'published_methods_v1/runs/internal' / f'budget_{budget:g}' / f'{model}_seed{seed}' / 'run.json'
                if not run_path.exists():
                    raise RuntimeError('Complete all-model/three-seed source cell required; missing ' + str(run_path))
                audit = audited[str(run_path.relative_to(ROOT))]
                assert sha256(ROOT / audit['verification']) == audit['sha256']
                bind(ROOT / audit['verification'])
                assert list(read_json(run_path)['inference_views']) == views
                for view in views:
                    suffix = '' if view == 'primary' else '_ensemble'
                    jobs.append({'system': model + '/' + view, 'seed': seed, 'run': run_path,
                                 'predictions': run_path.parent / f'validation_predictions{suffix}.npz',
                                 'view': view, 'published': True})
        for job in jobs:
            for path in [job['run'], job['predictions'], job['run'].parent / 'best.pt']:
                bind(path)
        cells.append({'budget': budget, 'dense_cell_id': cell['cell_id'], 'jobs': jobs})
    plan = {'version': 1, 'kind': 'Retrospective complete internal source comparison with full published-method ports',
            'analysis_code_sha256': sha256(__file__), 'seeds': SEEDS, 'budgets': [.2, 1.],
            'dense_families': DENSE_FAMILIES,
            'published_views': {'0.2': ['published_deeplab_fsl/primary', 'published_psmt_ssl/primary', 'published_psmt_ssl/ensemble'],
                                '1': ['published_deeplab_fsl/primary']},
            'endpoints': ENDPOINTS,
            'contrast_rule': 'Every dense system minus each applicable published view; SSL primary minus FSL primary; ensemble minus FSL primary and minus SSL primary. All three paired seeds and all six endpoints retained.',
            'analysis_timing': 'Retrospective: source outcomes already exist and UCF/SCDD development have informed study-wide design. This is not prospective or independent confirmation.',
            'statistics': 'Per-fit points, three-fit means and sample SD, all paired differences. No bootstrap, confidence intervals, p-values, adjusted significance, native-pixel pseudo-replication or across-budget replication claim.',
            'thresholds': 'Every view keeps its own original source logit threshold and temperature. Native cutoff0.5 remains unchanged.',
            'supervision': 'Match actual training/validation identities per fitted seed; SSL additionally uses remaining source-training images. UCF574 source/upstream HPO exposure remains additional. Equal masks do not equal objectives, optimizer work, pretraining or inference.',
            'timing': 'Raw timers are provenance only. Internal20% SSL seed22 includes a user pause; no timer is presented as isolated active compute. Teacher and ensemble views share one fit.',
            'input_bindings': sorted(bindings.values(), key=lambda x: x['path'])}
    plan_path = OUT / 'plan.json'
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise RuntimeError('Immutable analysis plan or inputs changed; preserve prior analysis and explicitly amend')
    else:
        save_json(plan_path, plan)
    identity = {'plan_sha256': sha256(plan_path)}
    directory = OUT / digest(identity)[:20]
    if (directory / 'outputs.json').exists():
        outputs = read_json(directory / 'outputs.json')
        for value in outputs['files']:
            assert sha256(ROOT / value['path']) == value['sha256']
        print(json.dumps({'reused': True, 'outputs': str((directory / 'outputs.json').relative_to(ROOT))}))
        return
    directory.mkdir(exist_ok=True)
    seed_rows = []
    comparisons = []
    cell_results = []
    for cell in cells:
        records = {}
        train_by_seed = {}
        annotation_reference = None
        truth_reference = None
        for job in cell['jobs']:
            run = read_json(job['run'])
            assert run['seed'] == job['seed'] and run['training_budget_fraction'] == cell['budget']
            assert sha256(job['run'].parent / 'best.pt') == run['checkpoint_sha256']
            assert not (job['run'].parent / 'latest.pt').exists()
            if job['published']:
                assert run['study_id'] == published_protocol['study_id']
                assert run['scientific_hashes'] == published_protocol['scientific_hashes']
                evaluated = run['inference_views'][job['view']]
            else:
                assert run['study_id'] == dense_protocol['study_id']
                evaluated = run
            training = sorted(run['training_ids'])
            if job['seed'] in train_by_seed:
                assert training == train_by_seed[job['seed']]
            else:
                train_by_seed[job['seed']] = training
            assert len(training) == run['training_images']
            assert len(training) + len(ids) == run['total_pixel_annotated_images_including_validation']
            assert not set(training) & set(ids)
            unlabelled = run.get('unlabelled_source_ids', [])
            assert not set(unlabelled) & (set(training) | set(ids))
            with np.load(job['predictions'], allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in ['ids', 'y', 'groups', 'logits']}
            view_run = {**run, 'source_logit_threshold': evaluated['source_logit_threshold']}
            record = saved_record(view_run, evaluated, arrays, job['system'], ids, groups)
            if truth_reference is None:
                truth_reference = record['y']
            assert np.array_equal(truth_reference, record['y'])
            annotation = [(x['id'], x['known_pixels'], x['crack_pixels'], x['negative_image_normal_structure_pixels'])
                          for x in evaluated['validation_localization']['per_image']]
            if annotation_reference is None:
                annotation_reference = annotation
            assert annotation == annotation_reference
            for key, value in classification(record['y'], record['logits'], record['source_threshold']).items():
                close(value, evaluated['validation_metrics'][key])
            for key, value in probability_metrics(record['y'], record['logits'], evaluated['source_temperature']).items():
                close(value, evaluated['validation_calibration_temperature_scaled'][key])
            reload = evaluated['checkpoint_reload']
            assert reload['fresh_model'] and max(reload['logit_max_abs_error'], reload['sampled_pixel_probability_max_abs_error']) <= 2e-5
            records.setdefault(job['system'], []).append(record)
            points = {name: endpoint_value(record, name) for name in ENDPOINTS}
            seed_rows.append({'budget': cell['budget'], 'system': job['system'], 'seed': job['seed'],
                              'run': str(job['run'].relative_to(ROOT)), 'view': job['view'],
                              'training_masks': len(training), 'validation_masks': len(ids),
                              'unlabelled_source_images': len(unlabelled), 'training_ID_set_sha256': digest(training),
                              'checkpoint_sha256': run['checkpoint_sha256'], 'source_threshold': record['source_threshold'],
                              'source_temperature': evaluated['source_temperature'], 'completed_rounds': run['completed_rounds'],
                              'optimizer_updates': run['optimizer_updates'],
                              'raw_training_validation_seconds_excluding_preparation': run['training_and_validation_seconds_excluding_preparation'],
                              'raw_timer_pause_affected': job['published'] and cell['budget'] == .2 and run['model']['semi'] and job['seed'] == 22,
                              'timer_is_isolated_active_compute': False, **points})
        for values in records.values():
            assert sorted(x['model']['seed'] for x in values) == SEEDS
        published_names = [x for x in records if x.startswith('published_')]
        contrast_pairs = [(dense, published) for dense in DENSE_FAMILIES for published in published_names]
        if cell['budget'] == .2:
            contrast_pairs += [('published_psmt_ssl/primary', 'published_deeplab_fsl/primary'),
                               ('published_psmt_ssl/ensemble', 'published_deeplab_fsl/primary'),
                               ('published_psmt_ssl/ensemble', 'published_psmt_ssl/primary')]
        for first, second in contrast_pairs:
            pairs = pair_records(records[first], records[second])
            endpoints = {}
            for endpoint in ENDPOINTS:
                points = [(endpoint_value(a, endpoint), endpoint_value(b, endpoint)) for a, b in pairs]
                differences = [a-b if a is not None and b is not None else None for a, b in points]
                endpoints[endpoint] = {'status': 'descriptive' if all(x is not None for x in differences) else 'not_estimable',
                                       'paired_seeds': SEEDS, 'first_values': [x[0] for x in points],
                                       'second_values': [x[1] for x in points], 'first_minus_second': summary(differences),
                                       'substitute_endpoint_selected': False}
            comparisons.append({'budget': cell['budget'], 'first': first, 'second': second, 'endpoints': endpoints})
        summaries = {name: {endpoint: summary([endpoint_value(x, endpoint) for x in values])
                           for endpoint in ENDPOINTS} for name, values in records.items()}
        cell_results.append({'budget': cell['budget'], 'source_validation_images': len(ids),
                             'positive_images': int(truth_reference.sum()), 'provided_capture_families': len(set(groups)),
                             'validation_ID_set_sha256': digest(ids), 'systems': summaries,
                             'unique_fits': len({str(x['run']) for x in cell['jobs']}),
                             'correlated_fit_views': len(cell['jobs'])})
    result = {'at': now(), 'identity': identity, 'plan': {'path': str(plan_path.relative_to(ROOT)), 'sha256': sha256(plan_path)},
              'complete_internal_cells': cell_results, 'all_seed_views': seed_rows, 'comparisons': comparisons,
              'unique_completed_fits_reused': len({x['run'] for x in seed_rows}), 'correlated_fit_views': len(seed_rows),
              'paired_contrasts': len(comparisons), 'endpoint_summaries': len(comparisons)*len(ENDPOINTS),
              'independent_confirmation': False, 'new_model_fit_inference_or_bootstrap': False,
              'scope': 'Same120 reused source-validation images in both annotation budgets. Three fitted seeds vary annotation subsets and stochastic training together. Additional unlabelled images, pretraining, source HPO, losses and update counts differ. No independent or matching-specific causal superiority or acceptance claim.'}
    save_json(directory / 'results.json', result)
    write_csv(directory / 'all_seed_views.csv', seed_rows)
    contrast_rows = []
    for comparison in comparisons:
        for endpoint, value in comparison['endpoints'].items():
            for index, seed in enumerate(SEEDS):
                contrast_rows.append({'budget': comparison['budget'], 'first': comparison['first'], 'second': comparison['second'],
                                      'endpoint': endpoint, 'seed': seed, 'first_value': value['first_values'][index],
                                      'second_value': value['second_values'][index], 'difference_raw_ratio': value['first_minus_second']['values'][index],
                                      'status': value['status']})
    write_csv(directory / 'all_paired_differences.csv', contrast_rows)
    tables = []
    for cell in cell_results:
        rows = []
        for system, values in cell['systems'].items():
            formatted = [('not estimable' if values[e]['mean'] is None else
                          f"{values[e]['mean']:.6f} +/- {values[e]['sample_SD']:.6f}") for e in ENDPOINTS]
            rows.append('<tr><th>'+html.escape(system)+'</th>'+''.join('<td>'+x+'</td>' for x in formatted)+'</tr>')
        tables.append('<h2>Internal training-mask fraction '+str(cell['budget'])+'</h2><div class="scroll"><table><tr><th>System/view</th>'+''.join('<th>'+html.escape(e)+'</th>' for e in ENDPOINTS)+'</tr>'+''.join(rows)+'</table></div>')
    report = '<!doctype html><meta charset="utf-8"><title>Complete internal published-method source comparisons</title><style>body{font:16px/1.5 system-ui;max-width:1500px;margin:30px auto;padding:20px;color:#183342}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:10px;border-bottom:1px solid #ccd7df;text-align:left;white-space:nowrap}</style><h1>Complete internal source comparisons, every declared model and three seeds</h1><p>Retrospective reused source validation; no independent confirmation. Mean plus/minus sample SD is not a confidence interval. All six endpoints use raw ratios. Native pooled pixel AP stays a whole-cohort point for each fit. The two budgets share the same120 validation images; views of the SSL teachers share a fit.</p><p>Every system retains its original checkpoint, source threshold, temperature and native0.5 cutoff. Training identities match within each fitted seed; SSL additionally uses remaining source-training images. Source/upstream UCF574 label exposure, different pretraining/losses/update counts and reused validation prevent an isolated backbone or matching-specific interpretation. High source recall is calibration-dependent.</p>'+''.join(tables)+'<p><a href="all_seed_views.csv">Every fitted seed/view and raw provenance</a> | <a href="all_paired_differences.csv">All31 applicable paired contrasts/six endpoints/three seeds</a> | <a href="results.json">Bound numerical result</a></p><p>Raw timers are provenance only; internal20% SSL seed22 was paused. No runtime average or speed comparison is inferred. Complete69-fit source/matched controls,97 isolated formal cost cases,24 ELPV outer/480inner fits and the one frozen independent confirmation remain required. Unsupported endpoints stay null; no favorable seed, alternative threshold or substitute endpoint is selected.</p>'
    (directory / 'report.html').write_text(report)
    files = [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in sorted(directory.iterdir())]
    outputs = {'at': now(), 'identity': identity, 'files': files, 'complete_cells': 2, 'new_inference': False}
    save_json(directory / 'outputs.json', outputs)
    save_json(OUT / 'latest.json', {'path': str((directory / 'results.json').relative_to(ROOT)), 'sha256': sha256(directory / 'results.json'),
                                  'outputs': {'path': str((directory / 'outputs.json').relative_to(ROOT)), 'sha256': sha256(directory / 'outputs.json')}})
    print(json.dumps({'result': str((directory / 'results.json').relative_to(ROOT)), 'fits_reused': result['unique_completed_fits_reused'],
                      'views': len(seed_rows), 'contrasts': len(comparisons), 'endpoint_summaries': result['endpoint_summaries']}))


if __name__ == '__main__':
    main()
