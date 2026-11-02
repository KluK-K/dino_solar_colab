"""All completed held-source dense cells: same checkpoints, two populations.

Saved full-float summaries only. No fitting/inference, score-based case search,
new statistical test or final-cohort access.
"""
import csv
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT, read_json, save_json, sha256, digest, now

BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'dense_transfer_readout_v1'
ARMS = [('deeplabv3plus_full', 'Adapted CNN', '#7c8e95'),
        ('dinov3_dense_frozen', 'DINO frozen', '#31869a'),
        ('dinov3_dense_lora', 'DINO LoRA', '#7955a8'),
        ('dinov3_dense_reference', 'Normal reference', '#b98135')]
ENDPOINTS = [('AUROC', 'Image AUROC'), ('AP', 'Native pixel AP'), ('Dice', 'Positive-image Dice')]


def main():
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Source-only readout must precede confirmation scoring')
    cells = [c for c in read_json(BASE / 'dense_development_complete_cells.json')['complete_cells'] if c['fold'] != 'internal']
    OUT.mkdir(exist_ok=True); pages = []
    for cell in cells:
        bindings = [{'path': cell['cell_path'], 'sha256': cell['cell_sha256']}]
        rows = []; source_support = None; target_support = None
        def support(local):
            return {'images': local['images'], 'positive_images': local['positive_images'],
                    'known_pixels': sum(p['known_pixels'] for p in local['per_image']),
                    'crack_pixels': sum(p['crack_pixels'] for p in local['per_image'])}
        for family, _, _ in ARMS:
            model = next(m for m in cell['models'] if m['family'] == family)
            if sorted(f['seed'] for f in model['fits']) != [11, 22, 33]: raise RuntimeError('Retain all fitted seeds')
            for fit in model['fits']:
                path = ROOT / fit['run']; run = read_json(path); target_path = path.parent / 'target_development.json'
                target = read_json(target_path)
                if sha256(path) != fit['run_sha256']: raise RuntimeError('Original source run changed')
                expected = next(r for r in cell['identity']['bindings'] if r['run'] == fit['run'])
                if sha256(target_path) != expected['evaluation_sha256']: raise RuntimeError('Original target result changed')
                bindings.extend([{'path': fit['run'], 'sha256': sha256(path)}, {'path': str(target_path.relative_to(ROOT)), 'sha256': sha256(target_path)}])
                a, b = support(run['validation_localization']), support(target['localization'])
                if source_support is None: source_support, target_support = a, b
                if (a, b) != (source_support, target_support): raise RuntimeError('Different annotation support across matched systems/seeds')
                for population, metrics, local in [('source', run['validation_metrics'], run['validation_localization']),
                                                   ('target', target['metrics'], target['localization'])]:
                    rows.append({'family': family, 'seed': fit['seed'], 'population': population,
                                 'training_masks': run['training_images'], 'validation_masks': run['validation_images'],
                                 'selected_round': run['best_round'], 'completed_rounds': run['completed_rounds'],
                                 'AUROC': metrics['auc'], 'AP': local['pooled_pixel_average_precision'],
                                 'Dice': local['mean_positive_image_dice'], 'recall': metrics['recall'], 'review': metrics['review_rate'],
                                 'predicted_crack_pixels': sum(p['predicted_crack_pixels'] for p in local['per_image']),
                                 'positive_images_with_empty_prediction': sum(p['crack_positive'] and p['predicted_crack_pixels'] == 0 for p in local['per_image'])})
        identity = {'bindings': bindings, 'generator_sha256': sha256(__file__),
                    'rule': 'All4 registered systems, all3 fitted seeds and both evaluation populations. Source-selected checkpoint fixed; descriptive cohort differences, not causal domain effects.'}
        directory = OUT / digest(identity)[:20]; directory.mkdir(exist_ok=True); result_path = directory / 'results.json'
        if not result_path.exists():
            cohort = cell['fold'].removeprefix('holdout_')
            style = {'font.family': 'DejaVu Sans', 'font.size': 7.2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#9baeb5', 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
            with plt.rc_context(matplotlib.rcParamsDefault), plt.rc_context(style):
                fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.3), sharex=True)
                for row_index, (key, ylabel) in enumerate(ENDPOINTS):
                    ceiling = 1. if key == 'AUROC' else min(1., max(.15, math.ceil(max(r[key] for r in rows) * 10) / 10 + .05))
                    for col, (family, label, color) in enumerate(ARMS):
                        ax = axes[row_index, col]; values = []
                        for seed, marker in [(11, 'o'), (22, '^'), (33, 's')]:
                            pair = [next(r[key] for r in rows if r['family'] == family and r['seed'] == seed and r['population'] == p) for p in ['source', 'target']]
                            if min(pair) < 0 or max(pair) > ceiling: raise RuntimeError('Do not clip source/target seed outcomes')
                            values.append(pair); ax.plot([0, 1], pair, color=color, alpha=.42, marker=marker, markersize=3.4, lw=.7)
                        ax.plot([0, 1], np.mean(values, axis=0), color=color, lw=1.8, marker='o', markersize=4)
                        ax.set_xlim(-.22, 1.22); ax.set_ylim(0, ceiling); ax.set_xticks([0, 1], ['Source', cohort])
                        ax.grid(axis='y', color='#e7edef', lw=.6); ax.set_axisbelow(True)
                        if key == 'AUROC': ax.axhline(.5, ls=':', lw=.7, color='#98a8ae')
                        if row_index == 0: ax.set_title(label, fontsize=8.2, pad=9)
                        if col == 0: ax.set_ylabel(ylabel, fontsize=7.4)
                fig.subplots_adjust(left=.095, right=.985, top=.85, bottom=.23, wspace=.5, hspace=.25)
                fig.suptitle(f"Held {cohort} · {100*cell['budget']:g}% source train masks · same checkpoints", fontsize=10, color='#284e5b', y=.98)
                fig.text(.095, .9, f"Source validation: {source_support['images']} images / {source_support['positive_images']} positives; held source: {target_support['images']} / {target_support['positive_images']}.", fontsize=7.2, color='#536f79')
                fig.text(.095, .145, 'Faint lines: all three fitted seeds (11 circle, 22 triangle, 33 square); thick lines: fitted-seed mean.', fontsize=6.8, color='#536f79')
                fig.text(.095, .1, 'Source validation selects checkpoints. Different populations, annotation footprints and pixel prevalences;', fontsize=6.8, color='#536f79')
                fig.text(.095, .065, 'the AP change is not an isolated causal domain effect. Curves are descriptive; no confidence interval.', fontsize=6.8, color='#536f79')
                masks = next(iter({r['training_masks'] for r in rows}))
                fig.text(.095, .025, f"{masks} source training masks + {source_support['images']} source validation masks + 574 upstream UCF calibration annotations.", fontsize=6.8, color='#536f79')
                for ext in ['png', 'pdf', 'svg']: fig.savefig(directory / f'matched_source_to_target.{ext}', dpi=300, facecolor='white')
                plt.close(fig)
            with (directory / 'all_fitted_seed_outcomes.csv').open('w') as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            save_json(result_path, {'at': now(), 'identity': identity, 'cell_id': cell['cell_id'], 'fold': cell['fold'], 'budget': cell['budget'],
                      'source_support': source_support, 'target_support': target_support, 'all_seed_population_records': rows,
                      'figures': [{'path': str((directory / f'matched_source_to_target.{ext}').relative_to(ROOT)), 'sha256': sha256(directory / f'matched_source_to_target.{ext}')} for ext in ['png', 'pdf', 'svg']],
                      'limits': ['Same source-selected fits on different development populations; not an independent test or a same-distribution causal estimate.',
                                 'All3 fitted-seed outcomes retained. Pooled AP depends on prevalence and annotation conventions, not just encoder transfer.',
                                 'This readout does not identify an optimizer or physical failure mechanism; source validation is not training-set fit.'],
                      'new_inference_or_training': False, 'final_confirmation_access': False})
        else:
            if read_json(result_path)['identity'] != identity: raise RuntimeError('Preserve previous readout snapshot')
        pages.append({'path': str(result_path.relative_to(ROOT)), 'sha256': sha256(result_path), 'fold': cell['fold'], 'budget': cell['budget']})
    save_json(OUT / 'index.json', {'at': now(), 'complete_held_source_cells': pages, 'planned_cells': 6, 'builder_sha256': sha256(__file__)})
    print('Same-checkpoint source/held-population readouts:', len(pages), 'complete cells; all4 systems and3 seeds')


if __name__ == '__main__': main()
