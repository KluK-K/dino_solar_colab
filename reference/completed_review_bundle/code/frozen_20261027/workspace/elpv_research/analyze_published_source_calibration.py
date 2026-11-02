"""Complete, metadata-only readout of the registered 18-fit calibration grid.

No models, prediction arrays, new fits, target scores, bootstrap or retuning.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import io
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'published_methods_v1'
OUT = BASE / 'published_source_calibration_readout_v1'
METRICS = ['image_AUROC', 'image_AP', 'native_pixel_AP', 'positive_Dice',
           'positive_IoU', 'image_recall', 'image_FPR', 'review_fraction',
           'negative_image_mean_pixel_FPR', 'normal_structure_FPR']


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'))
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')


def summarize(values):
    if any(v is None for v in values):
        return {'mean': None, 'sample_SD': None, 'values': values,
                'reason': 'At least one fitted system has an unsupported endpoint; no omission or zero imputation.'}
    assert all(math.isfinite(v) for v in values)
    return {'mean': statistics.mean(values), 'sample_SD': statistics.stdev(values), 'values': values}


def main():
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise RuntimeError('This retrospective source readout must precede confirmation')
    state = read(BASE / 'inference_efficiency_v1/status.json')['state']
    if not state.startswith('queued'):
        raise RuntimeError('Do not run during formal efficiency measurements')
    protocol_path = SOURCE / 'protocol.json'
    protocol = read(protocol_path)
    assert sha(ROOT / 'run_published_methods.py') == protocol['runner_sha256']
    for name, expected in protocol['scientific_hashes'].items():
        assert sha(ROOT / 'elresearch' / name) == expected, name
    assert protocol['calibration']['planned_runs'] == 18 and protocol['seeds'] == [11, 22, 33]
    index_path = BASE / 'published_output_integrity_v1/index.json'
    index_bytes = index_path.read_bytes()
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    audited = {r['run']: r for r in json.loads(index_bytes)['completed_published_fits_verified']}
    selection_path = SOURCE / 'selected_hyperparameters.json'
    selection = read(selection_path)
    plan = {'version': 1, 'analysis_code_sha256': sha(Path(__file__)),
            'kind': 'Retrospective complete source-calibration description',
            'source_observation': 'Written after all 18 source outcomes and registered selection exist; not prospective testing.',
            'requires': 'All six registered candidates, all three seeds and saved-output audit receipts.',
            'metrics': METRICS, 'primary_selection': protocol['calibration']['selection'],
            'views': 'Primary for every fit; declared same-checkpoint ensemble for SSL. Do not select views.',
            'uncertainty': 'Three fitted-system sample SD only; partial-budget seeds also change mask subsets. No CIs or significance tests.',
            'native_AP': 'Original full-cohort point per fitted system; no pixel resampling or sampled-to-native substitution.',
            'unsupported': 'Retain null; do not drop candidates or impute zero.',
            'limits': ['Same reused 144 validation images select checkpoint, operating points and rates.',
                       'SSL86 labelled +344 unlabelled and FSL430 labelled are unequal supervision conditions; all add144 validation masks and upstream UCF exposure.',
                       'Teacher/ensemble share fitting. This is not a complete matched DINO comparison, independent confirmation or reproduction of the paper score.',
                       'Preserve raw training timers; SIGSTOP contaminates the resumed seed33 elapsed value. No active-compute ranking.']}
    OUT.mkdir(exist_ok=True)
    snapshot = OUT / 'input_snapshots' / index_sha / 'published_output_integrity_index.json'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and snapshot.read_bytes() != index_bytes:
        raise RuntimeError('Preserve the immutable audit-index snapshot')
    if not snapshot.exists(): snapshot.write_bytes(index_bytes)
    plan_path = OUT / 'plan.json'
    if plan_path.exists():
        if read(plan_path) != plan: raise RuntimeError('Preserve the bound readout plan; amend explicitly')
    else: write(plan_path, plan)
    bindings = []
    def bind(p):
        item = {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
        if item not in bindings: bindings.append(item)
        return item
    for p in [Path(__file__), plan_path, protocol_path, selection_path, snapshot,
              ROOT / 'verify_completed_published_outputs.py']: bind(p)
    rows = []; candidates = []
    for spec in protocol['calibration']['candidates']:
        candidate_rows = []
        for seed in protocol['seeds']:
            path = SOURCE / 'calibration' / f"{spec['name']}_seed{seed}" / 'run.json'
            rel = str(path.relative_to(ROOT))
            if rel not in audited: raise RuntimeError('Incomplete/unaudited calibration: ' + rel)
            receipt = audited[rel]; proof_path = ROOT / receipt['verification']
            assert sha(proof_path) == receipt['sha256']
            run = read(path); history_path = path.parent / 'history.json'; history = read(history_path)
            assert run['study_id'] == protocol['study_id'] and run['seed'] == seed
            assert run['model']['name'] == spec['name'] and run['validation_images'] == 144
            assert run['validation_ids'] == protocol['source_validation_ids']
            assert set(run['training_ids'] + run['unlabelled_source_ids']) == set(protocol['source_training_ids'])
            source_proof = read(proof_path)['identity']['source_files']
            assert {'path': rel, 'sha256': sha(path)} in source_proof
            assert {'path': str((path.parent / 'best.pt').relative_to(ROOT)), 'sha256': run['checkpoint_sha256']} in source_proof
            assert (run['training_images'], run['unlabelled_source_images']) == ((86, 344) if spec['semi'] else (430, 0))
            assert len(history) == run['completed_rounds']
            assert max(history, key=lambda h: h['primary_source_selection_pixel_AP'])['round'] == run['best_round']
            run_binding = bind(path); bind(history_path); bind(proof_path)
            expected_views = ['primary', 'ensemble'] if spec['semi'] else ['primary']
            assert list(run['inference_views']) == expected_views
            for view in expected_views:
                data = run['inference_views'][view]; m = data['validation_metrics']; loc = data['validation_localization']
                assert m['n'] == 144 and m['positive'] == 58 and loc['positive_images'] == 58
                row = {'candidate': spec['name'], 'semi': spec['semi'], 'learning_rate': spec['learning_rate'],
                       'negative_threshold': spec.get('negative_threshold'), 'seed': seed, 'view': view,
                       'training_masks': run['training_images'], 'unlabelled_training_images': run['unlabelled_source_images'],
                       'validation_masks': 144, 'completed_rounds': run['completed_rounds'],
                       'optimizer_updates': run['optimizer_updates'], 'best_round': run['best_round'],
                       'checkpoint_sha256': run['checkpoint_sha256'], 'source_run': rel,
                       'source_run_sha256': run_binding['sha256'],
                       'raw_training_validation_seconds': run['training_and_validation_seconds_excluding_preparation'],
                       'known_SIGSTOP_contamination': spec['name'] == 'published_ssl_lr0.01_neg0.8' and seed == 33,
                       'image_AUROC': m['auc'], 'image_AP': m['average_precision'],
                       'native_pixel_AP': loc['pooled_pixel_average_precision'],
                       'positive_Dice': loc['mean_positive_image_dice'], 'positive_IoU': loc['mean_positive_image_iou'],
                       'image_recall': m['recall'], 'image_FPR': m['fp'] / (m['fp'] + m['tn']),
                       'review_fraction': m['review_rate'],
                       'negative_image_mean_pixel_FPR': loc['mean_negative_image_false_positive_fraction'],
                       'normal_structure_FPR': loc['pooled_negative_structure_false_positive_fraction']}
                rows.append(row); candidate_rows.append(row)
        summaries = {view: {metric: summarize([r[metric] for r in candidate_rows if r['view'] == view])
                            for metric in METRICS} for view in expected_views}
        candidates.append({'specification': spec, 'views': summaries,
                           'training_masks': 86 if spec['semi'] else 430,
                           'unlabelled_training_images': 344 if spec['semi'] else 0,
                           'validation_masks': 144})
    assert len(rows) == 30 and len(candidates) == 6
    expected_selection = []
    for semi in [False, True]:
        eligible = [c for c in candidates if c['specification']['semi'] == semi]
        chosen = max(eligible, key=lambda c: c['views']['primary']['native_pixel_AP']['mean'])
        expected_selection.append({**chosen['specification'], 'name': 'published_psmt_ssl' if semi else 'published_deeplab_fsl'})
    assert expected_selection == selection['selected'], 'Registered primary-native-AP selection disagrees'
    identity = {'analysis_sha256': sha(Path(__file__)), 'plan_sha256': sha(plan_path),
                'audit_index_observed_source': {'path': str(index_path.relative_to(ROOT)), 'sha256': index_sha,
                                              'immutable_snapshot': str(snapshot.relative_to(ROOT))},
                'input_bindings': bindings}
    target = OUT / hashlib.sha256(canonical(identity).encode()).hexdigest()[:20]; target.mkdir(exist_ok=True)
    csv_path = target / 'all18_fits_30_views.csv'
    result = {'at': now(), 'identity': identity, 'plan': plan, 'candidates': candidates,
              'rows': rows, 'verified_registered_selection': selection, 'fits': 18, 'inference_views': 30,
              'unique_source_validation_images': 144, 'source_positive_images': 58,
              'validation_used_for_selection': True, 'independent_confirmation': False,
              'new_model_load_fit_or_inference': False, 'new_hyperparameter_threshold_or_temperature_selection': False}
    result_path = target / 'results.json'
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode('utf-8')
    if csv_path.exists() and csv_path.read_bytes() != csv_bytes:
        raise RuntimeError('Preserve the immutable complete-source CSV')
    if result_path.exists():
        previous = read(result_path)
        if {k: v for k, v in previous.items() if k != 'at'} != {k: v for k, v in result.items() if k != 'at'}:
            raise RuntimeError('Preserve the immutable complete-source analysis')
    else:
        csv_path.write_bytes(csv_bytes)
        write(result_path, result)
    write(OUT / 'latest.json', {'at': now(), 'path': str(result_path.relative_to(ROOT)),
                              'sha256': sha(result_path), 'csv': {'path': str(csv_path.relative_to(ROOT)), 'sha256': sha(csv_path)}})
    print(json.dumps({'results': str(result_path.relative_to(ROOT)), 'fits': 18, 'views': 30,
                      'selection_verified': selection['selected'],
                      'normal_structure_FPR_estimable': any(r['normal_structure_FPR'] is not None for r in rows)}, indent=2))


if __name__ == '__main__': main()
