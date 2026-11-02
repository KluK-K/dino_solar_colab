"""Retrospective descriptive readout of all 97 completed formal timing jobs.

No models, checkpoint loading, inference, fitting, bootstrap or target access.
The original complete-evidence verifier has already checked the underlying files.
This new reporting closure binds its own source and every original result.
"""
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import statistics as st

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'inference_efficiency_v1'
OUT = BASE / 'complete_efficiency_readout_v1'
AUDIT = BASE / 'operational_wait_observations/formal_efficiency_completed_audit_2026-10-27T05-02-26.693101+00-00/summary.json'
EXPECTED_AUDIT = '55dcefb228b2bc76bbbcb2fcb6bf0cdbb2afaa6ad45cf3049b591662e13f7b73'
MEMORY = ['process_lifetime_highwater_RSS_bytes', 'mps_tensor_allocated_bytes', 'mps_driver_allocated_bytes']
METRICS = ['median_pipeline_ms_per_image', 'median_loop_ms_per_image', 'median_images_per_second', *MEMORY]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ref(p):
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def immutable(p, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists():
        assert p.read_text() == text, 'Bound reporting output changed: ' + str(p)
    else:
        p.write_text(text)


def summary(values):
    assert values and all(math.isfinite(x) for x in values)
    return {'n': len(values), 'values': values, 'mean': st.mean(values),
            'median': st.median(values), 'minimum': min(values), 'maximum': max(values),
            'sample_SD': st.stdev(values) if len(values) > 1 else None}


def csv_file(p, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with p.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, keys); writer.writeheader(); writer.writerows(rows)


def contrasts(families):
    pairs = []
    def add(a, b, reason):
        assert a in families and b in families
        pairs.append({'first': a, 'second': b, 'purpose': reason})
    for dino in ['frozen', 'lora', 'full']:
        for cnn in ['resnet18_full', 'resnet50_full', 'efficientnet_b0_full']:
            add('image__dinov3_' + dino, 'image__' + cnn, 'Image scoring implementation cost; no architecture-only causal claim')
    for baseline in ['frozen', 'full']:
        add('image__dinov3_lora', 'image__dinov3_' + baseline, 'Adaptation variant; fewer fitted parameters need not lower inference cost')
    for family in families:
        if family.startswith('dense__dinov3') and not family.endswith('_b20'):
            add(family, 'dense__deeplabv3plus_full', 'Dense image and native-map implementation cost')
    for a, b in [('dense__dinov3_dense_reference', 'dense__dinov3_dense_frozen'),
                 ('dense__dinov3_reference_no_position', 'dense__dinov3_dense_reference'),
                 ('dense__dinov3_reference_mean_evidence', 'dense__dinov3_dense_reference'),
                 ('mechanism__dinov3_postlora_reference', 'mechanism__dinov3_postlora_no_reference'),
                 ('mechanism__dinov3_postlora_no_reference', 'mechanism__dinov3_patch_mlp'),
                 ('published__ssl_primary', 'published__supervised_b20'),
                 ('published__ssl_ensemble', 'published__ssl_primary'),
                 ('published__supervised_b100', 'published__supervised_b20')]:
        add(a, b, 'Fixed source variant/view comparison; actual implementation and annotation exposure retained')
    for family in families:
        if family.endswith('_b20') and family.startswith(('dense__', 'mechanism__')):
            add(family, family[:-4], '20% versus full-budget reference-bank cost; shared available seeds only')
    return pairs


def main():
    assert not (BASE / 'final_confirmation_v1/observation_opened.json').exists()
    status = read(SOURCE / 'status.json')
    assert status['state'] == 'complete' and status['completed_jobs'] == status['planned_jobs'] == 97
    assert sha(AUDIT) == EXPECTED_AUDIT
    audit = read(AUDIT); assert audit['verified_jobs'] == 97
    for binding in audit['bindings'].values():
        assert sha(ROOT / binding['path']) == binding['sha256']
    protocol, verification = read(SOURCE / 'protocol.json'), read(SOURCE / 'verification.json')
    assert verification['verified_jobs'] == verification['planned_jobs'] == 97
    specs = {f"{m['family']}_seed{m['seed']}": m for m in protocol['models']}
    assert len(specs) == 97 and set(specs) == {x['id'] for x in verification['results']}
    families = list(dict.fromkeys(m['family'] for m in protocol['models']))
    plan = {'version': 1, 'kind': 'Complete retrospective source inference-cost reporting',
            'bindings': [ref(Path(__file__).absolute()), ref(AUDIT), ref(SOURCE / 'protocol.json'),
                         ref(SOURCE / 'model_bindings.json'), ref(SOURCE / 'verification.json'),
                         ref(ROOT / 'elresearch/efficiency_evidence.py')],
            'result_bindings': verification['results'], 'jobs': 97, 'families': families,
            'cohorts': list(protocol['cohorts']), 'batch_sizes': protocol['batch_sizes'],
            'contrasts': contrasts(families), 'metrics': METRICS,
            'aggregation': 'Each fit/condition retains all3 original uninstrumented passes. Main latency and throughput are medians of3 passes. All registered fitted seeds are shown; family means/sample SD and paired-seed differences/ratios are descriptive, not CIs. No pooling of repeats as independent fits.',
            'memory': 'Original separate component-pass observed counters in raw bytes. RSS highwater includes loading/warmup and earlier conditions in the same child. MPS tensor/driver counts are observed lower bounds, not guaranteed peaks or CUDA VRAM. Unified-memory pools overlap and must never be added.',
            'paired_policy': 'Intersect originally measured same-seed jobs. Full-budget global distance has only seed11, so its20%-full comparison has one pair; unmeasured seeds remain explicitly unpaired. A zero denominator makes a ratio null, not infinity.',
            'timing_scope': protocol['timing'], 'host': protocol['host'],
            'load_scope': 'Reported separately per job; process/import startup excluded and filesystem may be warm. Stored checkpoint bytes are not the whole deployed footprint.',
            'components': 'Separate synchronized instrumentation pass retained as hierarchy; component feature/classification/remaining values partition model-forward work and must not also be added to model_forward_and_transfers. They do not replace primary latency.',
            'equivalence_limit': 'Original verifier checks every image identity/logit and up to512 fixed uniform native probability samples/image across passes, not exact equality of every native pixel.',
            'interpretation': 'Same M2 Max32GiB/MPS host, already observed source inputs, registered model implementations. Read/checksum/decode/preprocess/forward/reference-search/transfers/decision/native reconstruction included; signature/profile/serialization/optimizer training excluded from primary times. No target accuracy, deployment SLA, cross-host speed or inference-speed inference from fitted-parameter counts.',
            'performance_cost_join': 'This closure reports cost only. Source task/budget/cohort/endpoint and actual annotation exposure must be matched explicitly before joining existing accuracy results; no cross-paper ranking or target-driven model selection.'}
    OUT.mkdir(exist_ok=True); immutable(OUT / 'plan.json', plan)
    destination = OUT / digest(plan)[:20]
    if destination.exists():
        outputs = read(destination / 'outputs.json')
        for binding in outputs['files']:
            assert sha(ROOT / binding['path']) == binding['sha256']
        print(json.dumps({'reused': True, 'result': ref(destination / 'results.json')})); return
    destination.mkdir()
    cases, passes, jobs = [], [], []
    for item in verification['results']:
        path = ROOT / item['result_path']; assert sha(path) == item['result_sha256']
        result, spec = read(path), specs[item['id']]
        assert result['model_id'] == item['id'] and result['prediction_equivalence_verified'] is True
        metadata = {key: value for key, value in result.items() if key not in ['cohorts', 'binding', 'numerical_signature_files']}
        jobs.append({'id': item['id'], 'family': spec['family'], 'seed': spec['seed'],
                     'kind': spec['kind'], 'source_view': spec['source_view'], 'metadata': metadata})
        for cohort in protocol['cohorts']:
            assert len(result['cohorts'][cohort]) == len(protocol['batch_sizes'])
            for case in result['cohorts'][cohort]:
                records = case['repeated_uninstrumented_passes']; assert len(records) == 3
                identity = {'model_id': item['id'], 'family': spec['family'], 'seed': spec['seed'],
                            'cohort': cohort, 'batch_size': case['batch_size'], 'input_size': case['input_size']}
                for n, repeat in enumerate(records):
                    assert repeat['images'] == len(protocol['cohorts'][cohort])
                    assert math.isclose(repeat['weighted_mean_ms_per_image'], repeat['timed_pipeline_seconds'] * 1000 / repeat['images'], rel_tol=1e-12)
                    passes.append({**identity, 'repeat': n + 1, **repeat})
                cases.append({**identity, 'median_pipeline_ms_per_image': st.median(x['weighted_mean_ms_per_image'] for x in records),
                              'median_loop_ms_per_image': st.median(x['observed_loop_seconds'] * 1000 / x['images'] for x in records),
                              'median_images_per_second': st.median(x['pipeline_images_per_second'] for x in records),
                              **case['memory']['maximum_observed'], 'memory_samples': case['memory']['samples'],
                              'pass_ms_per_image': [x['weighted_mean_ms_per_image'] for x in records],
                              'component_pass': case['component_pass'], 'prediction_equivalence': case['prediction_equivalence']})
    assert len(jobs) == 97 and len(cases) == 388 and len(passes) == 1164
    family_rows, paired_rows, contrast_rows = [], [], []
    for cohort in protocol['cohorts']:
        for batch in protocol['batch_sizes']:
            for family in families:
                points = sorted([x for x in cases if (x['family'], x['cohort'], x['batch_size']) == (family, cohort, batch)], key=lambda x: x['seed'])
                expected = sorted(x['seed'] for x in protocol['models'] if x['family'] == family)
                assert [x['seed'] for x in points] == expected
                family_rows.append({'family': family, 'cohort': cohort, 'batch_size': batch,
                                    'seeds': expected, 'metrics': {metric: summary([x[metric] for x in points]) for metric in METRICS}})
            for contrast in plan['contrasts']:
                points = {family: {x['seed']: x for x in cases if (x['family'], x['cohort'], x['batch_size']) == (family, cohort, batch)} for family in [contrast['first'], contrast['second']]}
                a, b = points[contrast['first']], points[contrast['second']]; common = sorted(a.keys() & b.keys())
                entry = {**contrast, 'cohort': cohort, 'batch_size': batch, 'paired_seeds': common,
                         'unpaired_first': sorted(a.keys() - b.keys()), 'unpaired_second': sorted(b.keys() - a.keys()), 'metrics': {}}
                for metric in METRICS:
                    values, ratios = [], []
                    for seed in common:
                        first, second = a[seed][metric], b[seed][metric]
                        difference = first - second; ratio = first / second if second != 0 else None
                        values.append(difference)
                        if ratio is not None: ratios.append(ratio)
                        paired_rows.append({**{k: entry[k] for k in ['first', 'second', 'cohort', 'batch_size']}, 'seed': seed, 'metric': metric, 'first_value': first, 'second_value': second, 'difference': difference, 'ratio': ratio})
                    entry['metrics'][metric] = {'difference': summary(values), 'ratio': summary(ratios) if ratios else None}
                contrast_rows.append(entry)
    result = {'plan': ref(OUT / 'plan.json'), 'jobs': jobs, 'cases': cases, 'family_summaries': family_rows,
              'paired_contrasts': contrast_rows, 'counts': {'jobs': len(jobs), 'families': len(families), 'conditions_per_job': 4, 'case_summaries': len(cases), 'uninstrumented_passes': len(passes), 'family_condition_summaries': len(family_rows), 'paired_contrast_conditions': len(contrast_rows), 'paired_metric_records': len(paired_rows)},
              'independent_confirmation': False, 'new_inference_fit_or_bootstrap': False}
    immutable(destination / 'results.json', result)
    csv_file(destination / 'all_fit_conditions.csv', [{k: v for k, v in x.items() if k not in ['component_pass', 'prediction_equivalence', 'pass_ms_per_image']} for x in cases])
    csv_file(destination / 'all_primary_passes.csv', [{k: v for k, v in x.items() if k not in ['stage_sum_seconds', 'batch_normalized_ms_per_image_percentile']} for x in passes])
    csv_file(destination / 'all_paired_metric_records.csv', paired_rows)
    sections = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete source inference-cost readout</title><style>body{font:15px system-ui;max-width:1600px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:6px;border:1px solid #ccc;text-align:right}td:first-child,th:first-child{text-align:left}thead{background:#eee}code{overflow-wrap:anywhere}</style><h1>All 97 formal source inference-cost jobs</h1>']
    sections.append('<p>Retrospective descriptive readout; all registered jobs, cohorts, batches and fitted seeds retained. Timings do not establish model accuracy. No final confirmation has been opened.</p><p>' + html.escape(plan['aggregation']) + '</p><p>' + html.escape(plan['memory']) + '</p><p>' + html.escape(plan['equivalence_limit']) + '</p>')
    for cohort in protocol['cohorts']:
        for batch in protocol['batch_sizes']:
            sections.append(f'<h2>{html.escape(cohort)}; batch {batch}</h2><p>Every cell reports the mean of the displayed fitted-seed values; brackets are min/max, not confidence intervals. Memory is MiB (2^20 bytes).</p><table><thead><tr><th>Family / fitted seeds</th><th>Pipeline ms/image</th><th>Images/s</th><th>RSS highwater MiB</th><th>MPS tensor MiB</th><th>MPS driver MiB</th></tr></thead><tbody>')
            for row in family_rows:
                if (row['cohort'], row['batch_size']) != (cohort, batch): continue
                fields = []
                for metric in ['median_pipeline_ms_per_image', 'median_images_per_second', *MEMORY]:
                    d = row['metrics'][metric]; scale = 2**20 if metric in MEMORY else 1
                    fields.append(f"<td>{d['mean']/scale:.3f} [{d['minimum']/scale:.3f}, {d['maximum']/scale:.3f}]</td>")
                sections.append('<tr><td>' + html.escape(row['family']) + ' / ' + str(row['seeds']) + '</td>' + ''.join(fields) + '</tr>')
            sections.append('</tbody></table>')
    sections.append('<h2>Loading and annotation/reference exposure</h2><p>Loading is a separate once-per-job observation; process/import startup excluded, filesystem may be warm. Stored checkpoint bytes exclude other dependencies and normal-reference objects. Zero PyTorch parameters for a classical estimator does not mean no learned coefficients.</p><table><tr><th>Job</th><th>Load seconds</th><th>Train / validation / unlabeled images</th><th>Reference images</th><th>Checkpoint MiB</th><th>Reference tensors / serialized MiB</th></tr>')
    for job in jobs:
        d = job['metadata']; sections.append(f"<tr><td>{html.escape(job['id'])}</td><td>{d['cold_model_and_bank_load_seconds']:.3f}</td><td>{d['model_source_training_images']} / {d['model_source_validation_images']} / {d['model_source_unlabelled_images']}</td><td>{d['normal_reference_images']}</td><td>{d['checkpoint_bytes']/2**20:.3f}</td><td>{d['normal_reference_tensor_storage_bytes']/2**20:.3f} / {d['normal_reference_serialized_bytes']/2**20:.3f}</td></tr>")
    sections.append('</table><h2>Scope and machine-readable evidence</h2><p>' + html.escape(plan['interpretation']) + '</p><p>' + html.escape(plan['performance_cost_join']) + '</p><p>All per-job/condition points, original repeat summaries, every declared paired contrast and null ratios are retained in results.json and the CSV files. No bootstrap, p-values or adjusted significance is reported. Figures and native manuscript integration remain separate tasks.</p></html>')
    (destination / 'report.html').write_text('\n'.join(sections))
    files = [x for x in destination.iterdir() if x.is_file()]
    immutable(destination / 'outputs.json', {'files': [ref(x) for x in sorted(files)]})
    immutable(OUT / 'latest.json', {'plan': ref(OUT / 'plan.json'), 'result': ref(destination / 'results.json'), 'outputs': ref(destination / 'outputs.json'), 'report': ref(destination / 'report.html'), 'counts': result['counts']})
    print(json.dumps({'result': ref(destination / 'results.json'), 'counts': result['counts']}))


if __name__ == '__main__':
    main()
