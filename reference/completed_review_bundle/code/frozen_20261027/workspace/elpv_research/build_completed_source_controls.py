"""Read completed source controls and saved scores; never fit or load a model."""
import csv
import hashlib
import html
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT, now, read_json, save_json, sha256
from elresearch.metrics import classification


BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'completed_source_controls_v1'
CNN_NAMES = {'resnet18': 'ResNet-18', 'resnet50': 'ResNet-50',
             'efficientnet_b0': 'EfficientNet-B0'}


def main():
    efficiency = read_json(BASE / 'inference_efficiency_v1/status.json')['state']
    if not (efficiency.startswith('queued') or efficiency in {'complete', 'failed', 'bounded_stop'}):
        raise RuntimeError('Do not plot or reanalyze scores during formal efficiency measurements')
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise RuntimeError('This descriptive report is confined to pre-confirmation development')
    cnn = BASE / 'cnn_stability_v1'; representation = BASE / 'representation_scale_v1'
    for folder, count in [(cnn, 18), (representation, 379)]:
        status = read_json(folder / 'status.json')
        if status['state'] != 'complete' or status['completed_runs'] != count:
            print('Require both complete original source-control stages; no partial ranking')
            return
    bindings = {}; checkpoints = []; verified_arrays = []

    def bind(path):
        path = Path(path)
        bindings[str(path.relative_to(ROOT))] = sha256(path)
        return read_json(path) if path.suffix == '.json' else None

    cp = bind(cnn / 'protocol.json'); rp = bind(representation / 'protocol.json')
    parent = BASE / 'scdd_development_v1'
    original = bind(parent / 'protocol.json')
    internal = next(f for f in original['split_manifest']['folds'] if f['name'] == 'internal')
    bind(cnn / 'status.json'); bind(representation / 'status.json')
    for folder, protocol, runner in [(cnn, cp, 'run_cnn_stability_controls.py'),
                                    (representation, rp, 'run_representation_scale_controls.py')]:
        if sha256(ROOT / runner) != protocol['runner_sha256']:
            raise RuntimeError('Registered scientific runner changed')
        for name, expected in protocol['scientific_hashes'].items():
            if sha256(ROOT / 'elresearch' / name) != expected:
                raise RuntimeError('Registered scientific dependency changed: ' + name)

    def check_predictions(path, expected_ids, metrics, threshold):
        bind(path)
        with np.load(path, allow_pickle=False) as z:
            if z['ids'].tolist() != expected_ids:
                raise RuntimeError('Saved prediction ordering differs: ' + str(path))
            actual = classification(z['y'], z['logits'], threshold)
            for key, expected in metrics.items():
                if expected is None:
                    if actual[key] is not None: raise RuntimeError('Null metric differs')
                elif not np.isclose(actual[key], expected, rtol=0, atol=2e-12):
                    raise RuntimeError('Saved prediction metric differs: ' + key)
            support = {'images': len(z['ids']), 'positive': int(z['y'].sum()),
                       'provided_capture_families': len(np.unique(z['groups']))}
        verified_arrays.append(str(path.relative_to(ROOT)))
        return support

    def checkpoint(path, expected):
        if sha256(path) != expected: raise RuntimeError('Saved checkpoint bytes differ')
        checkpoints.append({'path': str(path.relative_to(ROOT)), 'sha256': expected})

    cnn_runs = []; lora_runs = []
    for encoder in CNN_NAMES:
        for rate in [1e-5, 1e-4, 3e-4]:
            for seed in [11, 22, 33]:
                folder = (parent / 'runs/internal/budget_1' / f'{encoder}_full_seed{seed}'
                          if rate == 1e-4 else cnn / 'runs' / f'{encoder}_full_lr{rate:g}_seed{seed}')
                run = bind(folder / 'run.json'); receipt = bind(folder / 'source_reload_verification.json')
                if (run['seed'] != seed or run['model']['encoder_lr'] != rate or
                        run['validation_ids'] != internal['source_validation_ids'] or
                        set(run['training_ids']) != set(internal['source_train_ids']) or
                        (run['training_images'], run['validation_images']) != (351, 120)):
                    raise RuntimeError('CNN source-control identities differ')
                if (receipt['run_sha256'] != sha256(folder / 'run.json') or
                        receipt['checkpoint_sha256'] != run['checkpoint_sha256'] or
                        receipt['status'] != 'verified' or not receipt['auroc_exactly_preserved']):
                    raise RuntimeError('Original CNN fresh-reload receipt differs')
                checkpoint(folder / 'best.pt', run['checkpoint_sha256'])
                check_predictions(folder / 'validation_predictions.npz', run['validation_ids'],
                                  run['validation_metrics'], run['source_calibrated_logit_threshold'])
                cnn_runs.append({'encoder': encoder, 'learning_rate': rate, 'seed': seed,
                                 'AUROC': run['validation_metrics']['auc'],
                                 'image_AP': run['validation_metrics']['average_precision'],
                                 'best_round': run['best_round'], 'completed_rounds': run['completed_rounds'],
                                 'source_run': str((folder / 'run.json').relative_to(ROOT))})
    for seed in [11, 22, 33]:
        folder = parent / 'runs/internal/budget_1' / f'dinov3_lora_seed{seed}'
        run = bind(folder / 'run.json'); receipt = bind(folder / 'source_reload_verification.json')
        if receipt['run_sha256'] != sha256(folder / 'run.json') or receipt['status'] != 'verified':
            raise RuntimeError('Original image-only LoRA reload receipt differs')
        if run['validation_ids'] != internal['source_validation_ids']:
            raise RuntimeError('LoRA source-validation cohort differs')
        checkpoint(folder / 'best.pt', run['checkpoint_sha256'])
        check_predictions(folder / 'validation_predictions.npz', run['validation_ids'],
                          run['validation_metrics'], run['source_calibrated_logit_threshold'])
        lora_runs.append({'seed': seed, 'AUROC': run['validation_metrics']['auc'],
                          'source_run': str((folder / 'run.json').relative_to(ROOT))})
    cnn_summary = []
    for encoder in CNN_NAMES:
        for rate in [1e-5, 1e-4, 3e-4]:
            values = [r['AUROC'] for r in cnn_runs if (r['encoder'], r['learning_rate']) == (encoder, rate)]
            cnn_summary.append({'encoder': encoder, 'learning_rate': rate,
                                'mean_AUROC': statistics.mean(values), 'sample_SD': statistics.stdev(values)})

    trials = []
    for spec in rp['grid']:
        folder = representation / 'calibration_trials' / f"trial_{spec['candidate_index']:02d}"
        run = bind(folder / 'run.json'); trials.append(run)
        if run['specification'] != spec or run['training_ids'] != rp['source_training_ids']:
            raise RuntimeError('Calibration identity differs')
        if run['validation_ids'] != rp['source_validation_ids']:
            raise RuntimeError('Calibration validation identities differ')
        checkpoint(folder / 'model.joblib', run['checkpoint_sha256'])
        if not run['source_reload_verified'] or run['source_reload_maximum_absolute_error'] != 0:
            raise RuntimeError('Original fitted-probe reload verification failed')
        check_predictions(folder / 'validation_predictions.npz', run['validation_ids'],
                          run['validation_metrics'], run['source_calibrated_margin_threshold'])
    selection = bind(representation / 'selected_hyperparameters.json')
    families = [x['specification']['family'] for x in selection['selected']]
    if len(trials) != 49 or len(families) != 15 or len(set(families)) != 15:
        raise RuntimeError('Require all49 calibration trials and15 classifier families')
    for chosen in selection['selected']:
        candidates = [r for r in trials if r['specification']['family'] == chosen['specification']['family']]
        expected = max(candidates, key=lambda r: (r['validation_metrics']['auc'], -r['specification']['candidate_index']))
        if expected['specification'] != chosen['specification']:
            raise RuntimeError('Stored UCF-only hyperparameter selection differs')
        if sha256(ROOT / chosen['source_run_path'] / 'run.json') != chosen['source_run_sha256']:
            raise RuntimeError('Selected source run changed')
    rows = []; cells = []; training_sets = {}
    for fold in rp['split_manifest']['folds']:
        for budget in fold['budgets']:
            seeds = [11, 22, 33] if budget < 1 else [11]
            group = []
            for chosen in selection['selected']:
                spec = chosen['specification']
                for seed in seeds:
                    folder = representation / 'runs' / fold['name'] / f'budget_{budget:g}' / f"{spec['family']}_seed{seed}"
                    run = bind(folder / 'run.json')
                    if (run['study_id'] != rp['study_id'] or run['specification'] != spec or run['seed'] != seed or
                            run['validation_ids'] != fold['source_validation_ids'] or
                            not set(run['training_ids']).issubset(fold['source_train_ids'])):
                        raise RuntimeError('Development identity differs')
                    if budget == 1 and set(run['training_ids']) != set(fold['source_train_ids']):
                        raise RuntimeError('Full-budget source identities differ')
                    key = (fold['name'], budget, seed)
                    selected_ids = sorted(run['training_ids'])
                    if key in training_sets and training_sets[key] != selected_ids:
                        raise RuntimeError('Representations do not share annotation subsets')
                    training_sets[key] = selected_ids
                    if not run['source_reload_verified'] or run['source_reload_maximum_absolute_error'] != 0:
                        raise RuntimeError('Original probe reload verification failed')
                    checkpoint(folder / 'model.joblib', run['checkpoint_sha256'])
                    support = check_predictions(folder / 'validation_predictions.npz', run['validation_ids'],
                                                run['validation_metrics'], run['source_calibrated_margin_threshold'])
                    metrics = run['validation_metrics']
                    if fold['name'] != 'internal':
                        target = bind(folder / 'target_development.json')
                        if (target['threshold_fitted_on_target'] or target['checkpoint_sha256'] != run['checkpoint_sha256'] or
                                target['target_ids'] != fold['target_development_ids']):
                            raise RuntimeError('Held-source selection or cohort differs')
                        if set(target['target_ids']) & set(run['training_ids'] + run['validation_ids']):
                            raise RuntimeError('Held-source identity leakage')
                        metrics = target['metrics']
                        support = check_predictions(folder / 'target_development_predictions.npz', target['target_ids'],
                                                    metrics, run['source_calibrated_margin_threshold'])
                    row = {'fold': fold['name'], 'budget': budget, 'family': spec['family'], 'seed': seed,
                           'training_images': run['training_images'], 'validation_images': run['validation_images'],
                           'AUROC': metrics['auc'], 'image_AP': metrics['average_precision'],
                           'recall': metrics['recall'], 'review_fraction': metrics['review_rate'],
                           'provided_capture_families': support['provided_capture_families'],
                           'evaluation_images': support['images'], 'positive_images': support['positive'],
                           'source_run': str((folder / 'run.json').relative_to(ROOT))}
                    rows.append(row); group.append(row)
            if len(group) != 15 * len(seeds): raise RuntimeError('Incomplete family/seed cell')
            summaries = []
            for family in families:
                rr = [r for r in group if r['family'] == family]
                summaries.append({'family': family, 'fits': len(rr), 'mean_AUROC': statistics.mean(r['AUROC'] for r in rr),
                                  'sample_SD': statistics.stdev(r['AUROC'] for r in rr) if len(rr) > 1 else None})
            cells.append({'fold': fold['name'], 'budget': budget, 'families': summaries,
                          'evaluation_images': group[0]['evaluation_images'], 'positive_images': group[0]['positive_images'],
                          'training_images': group[0]['training_images'], 'validation_images': group[0]['validation_images']})
    if len(rows) != 330 or len(cells) != 10 or len(checkpoints) != 409:
        raise RuntimeError('Require27 CNN +3 LoRA reference +379 representation fits')
    identity = {'builder_sha256': sha256(__file__), 'input_bindings': [{'path': p, 'sha256': h} for p, h in sorted(bindings.items())],
                'scope': 'Descriptive completed source/development controls, no new fitting, model inference, thresholds or confidence intervals'}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    folder = OUT / key
    if (folder / 'results.json').exists():
        old = read_json(folder / 'results.json')
        if old['identity'] != identity: raise RuntimeError('Preserve original descriptive report')
        for asset in old['figures']:
            if sha256(ROOT / asset['path']) != asset['sha256']: raise RuntimeError('Preserve original figure bytes')
        save_json(OUT / 'latest.json', {'path': str((folder / 'results.json').relative_to(ROOT)), 'sha256': sha256(folder / 'results.json')})
        print('Reused complete source controls', folder); return
    folder.mkdir(parents=True, exist_ok=False)
    assets = []
    style = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
             'pdf.fonttype': 42, 'svg.fonttype': 'none'}

    def save(fig, stem):
        for ext in ['png', 'pdf', 'svg']:
            path = folder / (stem + '.' + ext); fig.savefig(path, dpi=240, facecolor='white')
            assets.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
        plt.close(fig)

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.2), sharey=True)
        fig.subplots_adjust(left=.09, right=.98, top=.74, bottom=.37, wspace=.14)
        lora_mean = statistics.mean(r['AUROC'] for r in lora_runs)
        for ax, (encoder, name) in zip(axes, CNN_NAMES.items()):
            summary = [x for x in cnn_summary if x['encoder'] == encoder]
            ax.errorbar([r['learning_rate'] for r in summary], [r['mean_AUROC'] for r in summary],
                        yerr=[r['sample_SD'] for r in summary], fmt='D-', color='#27677a', ms=4, lw=1.1, capsize=3)
            for seed, marker in zip([11, 22, 33], ['o', '^', 's']):
                rr = [r for r in cnn_runs if r['encoder'] == encoder and r['seed'] == seed]
                ax.scatter([r['learning_rate'] for r in rr], [r['AUROC'] for r in rr], marker=marker, s=16, color='#27677a', alpha=.4)
            ax.axhline(lora_mean, color='#8057b5', ls='--', lw=1)
            ax.set_xscale('log'); ax.set_xlim(6e-6, 5e-4); ax.set_ylim(.45, 1.01)
            ax.set_xticks([1e-5, 1e-4, 3e-4], ['1e-5', '1e-4', '3e-4']); ax.minorticks_off()
            ax.set_xlabel('Encoder learning rate'); ax.set_title(name, loc='left'); ax.grid(axis='y', color='#e5ecef')
            for side in ['top', 'right']: ax.spines[side].set_visible(False)
        axes[0].set_ylabel('Source-validation image AUROC')
        fig.text(.09, .93, 'CNN optimization sensitivity: all 27 original source fits', fontsize=11, weight='bold', color='#183f4d')
        fig.text(.09, .86, '351 training images + 120 reused validation images; image labels only', fontsize=8)
        notes = ['Diamonds/bars: three-seed mean ± sample SD. Faint points retain every original fitted seed.',
                 '1e-4: original CNN setting; 1e-5 and 3e-4: post-hoc registered sensitivity. All weaker settings remain.',
                 f'Purple dashed: original image-only DINOv3 LoRA mean ({lora_mean:.3f}); not the joint dense decoder.',
                 'Validation chooses checkpoints and informed these controls. These are optimistic development summaries.',
                 'Different HPO histories; no independent or transfer superiority follows. No new model inference.']
        for i, note in enumerate(notes): fig.text(.06, .24-i*.041, note, fontsize=6.6, color='#526d79')
        save(fig, 'cnn_learning_rate_controls')

        labels = []
        for family in families:
            spec = next(x['specification'] for x in selection['selected'] if x['specification']['family'] == family)
            encoder = {'dinov3_small': 'DINOv3-S', 'dinov3_base': 'DINOv3-B', 'resnet18': 'ResNet-18'}[spec['encoder']]
            pool = {'cls': 'CLS', 'patch_mean': 'patch mean', 'concat': 'CLS + patch', 'global': 'global'}[spec['pooling']]
            classifier = {'logistic': 'LR', 'linear_svm': 'linear SVM', 'rbf_svm': 'RBF SVM'}[spec['classifier']]
            labels.append(f"{encoder} / {spec['input_size']} / {pool} / {classifier}")
        for internal_only, stem in [(True, 'representation_internal'), (False, 'representation_held_sources')]:
            selected_cells = [c for c in cells if (c['fold'] == 'internal') == internal_only]
            values = np.array([[c['families'][i]['mean_AUROC'] for c in selected_cells] for i in range(15)])
            fig, ax = plt.subplots(figsize=(8.3, 7.5)); fig.subplots_adjust(left=.37, right=.94, top=.80, bottom=.24)
            image = ax.imshow(values, vmin=0, vmax=1, cmap='cividis', aspect='auto')
            ax.set_yticks(range(15), labels); ax.tick_params(axis='both', length=0, labelsize=7)
            columns = [((f"{100*c['budget']:g}%" if internal_only else f"{c['fold'].removeprefix('holdout_')}\n{100*c['budget']:g}%") +
                        ('\n3 subsets' if c['budget'] < 1 else '\n1 fit')) for c in selected_cells]
            ax.set_xticks(range(len(columns)), columns); ax.xaxis.tick_top(); ax.tick_params(axis='x', pad=8)
            for y in range(15):
                for x, cell in enumerate(selected_cells):
                    row = cell['families'][y]; sd = row['sample_SD']
                    label = f"{row['mean_AUROC']:.3f}" + (f"\n±{sd:.3f}" if sd is not None else '')
                    ax.text(x, y, label, ha='center', va='center', fontsize=6.8, color='white' if values[y, x] < .58 else '#111111')
            for boundary in [2.5, 5.5, 8.5, 11.5]: ax.axhline(boundary, color='white', lw=1.3)
            for side in ax.spines.values(): side.set_visible(False)
            title = 'Frozen representations: internal source validation' if internal_only else 'Frozen representations: every held-source development cell'
            fig.text(.06, .95, title, fontsize=11, weight='bold', color='#183f4d')
            fig.text(.06, .905, '15 classifier families / 13 representations; all 330 unique development fits retained', fontsize=8)
            bar = fig.colorbar(image, ax=ax, orientation='horizontal', fraction=.04, pad=.04)
            bar.set_label('Image AUROC · shared 0–1 scale', fontsize=7); bar.ax.tick_params(labelsize=7)
            notes = ['Cells: mean ± sample SD across three training-image subsets below 100%; one deterministic fit at 100%.',
                     'LR = logistic regression. All 13 LR representations share the same three-C UCF-only selection grid.',
                     'UCF 430 training + 144 validation labels inform selection; additional SCDD validation labels set thresholds.',
                     ('The same 120 source-validation images are reused across four budgets; no independent confirmation or CI.' if internal_only else 'Held panels reuse three acquisition cohorts across two budgets. They are observed development, not tests.'),
                     'Processing resolution does not recover missing physical detail. No target-fitted scaler or new inference.']
            for i, note in enumerate(notes): fig.text(.06, .16-i*.027, note, fontsize=6.7, color='#526d79')
            save(fig, stem)

    csv_bindings = []
    for stem, data in [('all27_cnn_source_fits', cnn_runs), ('all330_representation_development_fits', rows)]:
        path = folder / (stem + '.csv')
        with path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
        csv_bindings.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
    result = {'at': now(), 'identity': identity, 'CNN_runs': cnn_runs, 'CNN_summary': cnn_summary,
              'image_LoRA_reference_runs': lora_runs, 'image_LoRA_mean_AUROC': lora_mean,
              'representation_runs': rows, 'representation_cells': cells, 'family_order': families,
              'selection': selection, 'checkpoint_bindings_verified': checkpoints,
              'saved_prediction_arrays_recomputed': verified_arrays,
              'counts': {'CNN_sensitivity': 18, 'original_CNN': 9, 'original_image_LoRA_reference': 3,
                         'representation_calibration': 49, 'representation_development': 330,
                         'checkpoints': len(checkpoints), 'saved_prediction_arrays': len(verified_arrays)},
              'figures': assets, 'csv': csv_bindings, 'new_model_fitting': False,
              'new_model_inference': False, 'new_confidence_intervals': False, 'final_confirmation_access': False,
              'limits': ['Descriptive post-hoc source controls; no independent comparison or physical mechanism.',
                         'Source calibration/HPO exposure is additional to SCDD annotation budgets.',
                         'CNN sensitivity concerns image-only models, not an optimized dense-CNN transfer experiment.',
                         'Deterministic full-budget probe fits are saved once; three lower-budget seeds vary subsets only.']}
    save_json(folder / 'results.json', result)
    save_json(OUT / 'latest.json', {'path': str((folder / 'results.json').relative_to(ROOT)), 'sha256': sha256(folder / 'results.json')})
    cards = ''.join(f'<figure><img src="{Path(a["path"]).name}"></figure>' for a in assets if a['path'].endswith('.png'))
    table = ''.join(f'<tr><td>{CNN_NAMES[r["encoder"]]}</td><td>{r["learning_rate"]:g}</td><td>{r["mean_AUROC"]:.3f} ± {r["sample_SD"]:.3f}</td></tr>' for r in cnn_summary)
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Completed source controls</title>
<style>body{{max-width:1100px;margin:48px auto;font:17px/1.55 system-ui;color:#193c4a}}img{{width:100%}}figure{{margin:32px 0}}td,th{{padding:8px 24px;border-bottom:1px solid #dce7ec}}.note{{background:#edf4f6;padding:18px}}a{{color:#25677b}}</style>
<h1>Completed source controls</h1><p class="note">All 27 CNN source fits, 3 original image-only LoRA references, 49 representation calibration trials and 330 unique representation development fits. All 409 saved checkpoint hashes and{len(verified_arrays)} saved prediction arrays checked. No new model fit, inference, threshold or independent test.</p>
<p>The lower CNN learning rate changes source-validation conclusions. This is an image-only optimization sensitivity, not an independent test or evidence that the dense CNN transfer result has been reversed. Frozen-feature matrices retain every family, cohort, budget and registered subset.</p>
<table><tr><th>Image-only CNN</th><th>Encoder LR</th><th>Source AUROC ± sample SD</th></tr>{table}</table>{cards}
<p>Reference image-only DINOv3 LoRA mean source AUROC: {lora_mean:.3f}. The same 120 validation images selected checkpoints and motivated the sensitivity. No source-selected winner is promoted to an independent superiority claim.</p>
<p><a href="results.json">All values and input hashes</a> · <a href="all27_cnn_source_fits.csv">Every CNN fit</a> · <a href="all330_representation_development_fits.csv">Every representation development fit</a></p></html>'''
    (folder / 'report.html').write_text(report)
    print('Complete descriptive source controls', folder)
    print('Counts', json.dumps(result['counts']))


if __name__ == '__main__':
    main()
