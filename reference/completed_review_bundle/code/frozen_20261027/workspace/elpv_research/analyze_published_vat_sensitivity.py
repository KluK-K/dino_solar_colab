"""Read the complete registered source VAT contrast from audited saved metadata.

No model imports, prediction arrays, inference, resampling or source retuning.
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
OUT = BASE / 'published_vat_sensitivity_readout_v1'
METRICS = ['image_AUROC', 'image_AP', 'native_pixel_AP', 'positive_Dice',
           'positive_IoU', 'image_recall', 'image_FPR', 'review_fraction',
           'negative_image_mean_pixel_FPR', 'normal_structure_FPR']


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'))
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p, v): p.write_text(json.dumps(v, indent=2) + '\n')


def summary(values):
    if any(v is None for v in values):
        return {'values': values, 'mean': None, 'sample_SD': None,
                'reason': 'Unsupported endpoint retained; no omission or zero imputation.'}
    assert len(values) == 3 and all(math.isfinite(v) for v in values)
    return {'values': values, 'mean': statistics.mean(values),
            'sample_SD': statistics.stdev(values)}


def main():
    assert not (BASE / 'final_confirmation_v1/observation_opened.json').exists()
    assert read(BASE / 'inference_efficiency_v1/status.json')['state'].startswith('queued')
    pp = SOURCE / 'protocol.json'; protocol = read(pp)
    assert sha(ROOT / 'run_published_methods.py') == protocol['runner_sha256']
    for name, value in protocol['scientific_hashes'].items():
        assert sha(ROOT / 'elresearch' / name) == value, name
    seeds = protocol['seeds']; assert seeds == [11, 22, 33]
    literal = protocol['calibration']['literal_recipe_sensitivity']
    assert literal['runs'] == 3 and literal['vat_xi'] == 1e-6
    control = next(c for c in protocol['calibration']['candidates']
                   if c['semi'] and c['learning_rate'] == literal['learning_rate']
                   and c['negative_threshold'] == literal['negative_threshold'])
    assert control['name'] == 'published_ssl_lr0.01_neg0' and control['vat_xi'] == .1
    index_path = BASE / 'published_output_integrity_v1/index.json'
    index_bytes = index_path.read_bytes(); index_sha = hashlib.sha256(index_bytes).hexdigest()
    audited = {v['run']: v for v in json.loads(index_bytes)['completed_published_fits_verified']}
    OUT.mkdir(exist_ok=True)
    snap = OUT / 'input_snapshots' / index_sha / 'published_output_integrity_index.json'
    snap.parent.mkdir(parents=True, exist_ok=True)
    if snap.exists(): assert snap.read_bytes() == index_bytes
    else: snap.write_bytes(index_bytes)
    plan = {'version': 1, 'analysis_code_sha256': sha(Path(__file__)),
            'kind': 'Retrospective complete readout of a prospectively registered source-only numerical contrast',
            'analysis_written_after_outcomes': True,
            'contrast': 'Released-caller xi1e-6 minus canonical FP32 xi0.1 at the SAME lr0.01 and negative threshold0.',
            'requires': 'Both settings, all three seeds, all declared primary/ensemble views and saved-output audits.',
            'metrics': METRICS, 'uncertainty': 'All paired seeds and mean/sample SD only; no CIs or adjusted tests.',
            'native_AP': 'Original whole-cohort native point per fitted system; no pixel bootstrap.',
            'selection': 'No new selection. Never compare with the chosen lr0.001 as an isolated xi contrast.',
            'limits': ['Same reused144 source-validation images choose checkpoints and image operating points.',
                       'Within-seed source subsets/initialization are checked; early stopping and selected rounds can differ.',
                       'Across seeds, the mask subset and stochastic fit vary together.',
                       'Two inference views share one fit; absent normal-structure support remains null.',
                       'Local FP32/source sensitivity does not establish failure of author training on other hardware.',
                       'No independent superiority, cross-paper score ranking or formal speed comparison.']}
    plan_path = OUT / 'plan.json'
    if plan_path.exists(): assert read(plan_path) == plan, 'Preserve the bound analysis source/plan'
    else: write(plan_path, plan)
    bindings = []
    def bind(p):
        b = {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
        if b not in bindings: bindings.append(b)
        return b
    for p in [Path(__file__), plan_path, pp, snap, ROOT / 'verify_completed_published_outputs.py',
              SOURCE / 'selected_hyperparameters.json']: bind(p)
    rows = []; supports = []
    for seed in seeds:
        pair = []
        for condition, folder, name, xi in [
            ('canonical', 'calibration', control['name'], .1),
            ('literal', 'source_sensitivity', 'published_ssl_literal_xi1e6', 1e-6)]:
            p = SOURCE / folder / f'{name}_seed{seed}' / 'run.json'
            rel = str(p.relative_to(ROOT)); assert rel in audited, rel
            receipt = ROOT / audited[rel]['verification']
            assert sha(receipt) == audited[rel]['sha256']
            run = read(p); proof = read(receipt)
            assert run['study_id'] == protocol['study_id'] and run['seed'] == seed
            assert run['model']['semi'] and run['model']['vat_xi'] == xi
            assert run['model']['learning_rate'] == .01 and run['model']['negative_threshold'] == 0
            assert run['validation_ids'] == protocol['source_validation_ids']
            assert (run['training_images'], run['unlabelled_source_images'], run['validation_images']) == (86, 344, 144)
            assert set(run['training_ids'] + run['unlabelled_source_ids']) == set(protocol['source_training_ids'])
            assert {'path': rel, 'sha256': sha(p)} in proof['identity']['source_files']
            assert {'path': str((p.parent / 'best.pt').relative_to(ROOT)),
                    'sha256': run['checkpoint_sha256']} in proof['identity']['source_files']
            hp = p.parent / 'history.json'; history = read(hp)
            assert len(history) == run['completed_rounds']
            assert max(history, key=lambda x: x['primary_source_selection_pixel_AP'])['round'] == run['best_round']
            for f in [p, hp, receipt]: bind(f)
            assert list(run['inference_views']) == ['primary', 'ensemble']
            for view, v in run['inference_views'].items():
                m = v['validation_metrics']; loc = v['validation_localization']
                assert m['n'] == 144 and m['positive'] == loc['positive_images'] == 58
                rows.append({'condition': condition, 'candidate': name, 'vat_xi': xi,
                             'learning_rate': .01, 'negative_threshold': 0, 'seed': seed, 'view': view,
                             'training_masks': 86, 'unlabelled_training_images': 344, 'validation_masks': 144,
                             'completed_rounds': run['completed_rounds'], 'optimizer_updates': run['optimizer_updates'],
                             'best_round': run['best_round'], 'checkpoint_sha256': run['checkpoint_sha256'],
                             'source_run': rel, 'source_run_sha256': sha(p),
                             'raw_training_validation_seconds': run['training_and_validation_seconds_excluding_preparation'],
                             'image_AUROC': m['auc'], 'image_AP': m['average_precision'],
                             'native_pixel_AP': loc['pooled_pixel_average_precision'],
                             'positive_Dice': loc['mean_positive_image_dice'],
                             'positive_IoU': loc['mean_positive_image_iou'], 'image_recall': m['recall'],
                             'image_FPR': m['fp'] / (m['fp'] + m['tn']), 'review_fraction': m['review_rate'],
                             'negative_image_mean_pixel_FPR': loc['mean_negative_image_false_positive_fraction'],
                             'normal_structure_FPR': loc['pooled_negative_structure_false_positive_fraction']})
            pair.append(run)
        a, b = pair
        for k in ['training_ids', 'unlabelled_source_ids', 'validation_ids', 'input_size',
                  'updates_per_round', 'maximum_rounds', 'initialization']:
            assert a[k] == b[k], (seed, k)
        for k in ['backbone_sha256', 'archive_basis', 'checkpoint_format']:
            assert a['model'][k] == b['model'][k], (seed, k)
        supports.append({'seed': seed, 'identical_source_subsets_and_initialization': True,
                         'training_ids_sha256': hashlib.sha256(canonical(a['training_ids']).encode()).hexdigest(),
                         'unlabelled_ids_sha256': hashlib.sha256(canonical(a['unlabelled_source_ids']).encode()).hexdigest(),
                         'validation_ids_sha256': hashlib.sha256(canonical(a['validation_ids']).encode()).hexdigest(),
                         'initialization_sha256': hashlib.sha256(canonical(a['initialization']).encode()).hexdigest()})
    assert len(rows) == 12
    paired = []
    for view in ['primary', 'ensemble']:
        for seed in seeds:
            a = next(r for r in rows if r['condition'] == 'canonical' and r['seed'] == seed and r['view'] == view)
            b = next(r for r in rows if r['condition'] == 'literal' and r['seed'] == seed and r['view'] == view)
            item = {'view': view, 'seed': seed}
            for metric in METRICS:
                item[metric] = None if a[metric] is None or b[metric] is None else b[metric] - a[metric]
            paired.append(item)
    summaries = {condition: {view: {metric: summary([next(r for r in rows if r['condition'] == condition
                 and r['view'] == view and r['seed'] == seed)[metric] for seed in seeds])
                 for metric in METRICS} for view in ['primary', 'ensemble']} for condition in ['canonical', 'literal']}
    differences = {view: {metric: summary([next(r for r in paired if r['view'] == view and r['seed'] == seed)[metric]
                   for seed in seeds]) for metric in METRICS} for view in ['primary', 'ensemble']}
    identity = {'analysis_code_sha256': sha(Path(__file__)), 'input_bindings': bindings}
    target = OUT / hashlib.sha256(canonical(identity).encode()).hexdigest()[:20]
    target.mkdir(exist_ok=True)
    result = {'at': now(), 'identity': identity, 'plan': plan, 'fits': 6, 'new_literal_fits': 3,
              'inference_views': 12, 'unique_source_validation_images': 144, 'source_positive_images': 58,
              'paired_support_checks': supports, 'rows': rows, 'paired_literal_minus_canonical': paired,
              'summaries': summaries, 'paired_differences': differences,
              'registered_source_selection_unchanged': read(SOURCE / 'selected_hyperparameters.json'),
              'independent_confirmation': False, 'new_model_load_fit_or_inference': False,
              'new_hyperparameter_threshold_or_temperature_selection': False}
    rp = target / 'results.json'
    if rp.exists():
        prior = read(rp); result['at'] = prior['at']; assert prior == result
    else: write(rp, result)
    csvs = []
    for name, records in [('all6_fits_12_views.csv', rows), ('all6_paired_view_differences.csv', paired)]:
        buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        p = target / name; value = buf.getvalue().encode()
        if p.exists(): assert p.read_bytes() == value
        else: p.write_bytes(value)
        csvs.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p)})
    latest = {'at': now(), 'path': str(rp.relative_to(ROOT)), 'sha256': sha(rp), 'csvs': csvs}
    write(OUT / 'latest.json', latest)
    print(json.dumps({'result': latest, 'primary_summary': {k: summaries[k]['primary'] for k in summaries},
                      'primary_paired_differences': differences['primary']}, indent=2))


if __name__ == '__main__': main()
