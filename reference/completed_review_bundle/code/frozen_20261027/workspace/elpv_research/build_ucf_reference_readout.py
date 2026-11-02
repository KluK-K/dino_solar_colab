"""Descriptive readout of all original UCF source controls; no model inference."""
import csv
import hashlib
import json
import statistics

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from elresearch.common import ROOT, now, read_json, save_json, sha256


BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'ucf_reference_transfer_v1'
PARENT = BASE / 'dense_development_v1'
OUT = BASE / 'ucf_reference_readout_v1'
LABELS = {
    'deeplabv3plus_full': 'Adapted CNN',
    'dinov3_dense_frozen': 'DINOv3 frozen',
    'dinov3_dense_lora': 'DINOv3 LoRA',
    'dinov3_dense_full': 'DINOv3 full',
    'ucf_reference_position': 'Reference + position',
    'ucf_reference_no_position': 'Reference, no position',
    'ucf_reference_mean_evidence': 'Reference, mean evidence',
}
METRICS = [('image_AUROC', 'Image AUROC'), ('native_pixel_AP', 'Native pixel AP'),
           ('positive_image_Dice', 'Positive-image Dice')]


def main():
    state = read_json(SOURCE / 'status.json')
    if state['state'] != 'complete' or state['completed_runs'] != 9:
        print('No partial UCF reference comparison: all9 original source fits required')
        return
    protocol_path = SOURCE / 'protocol.json'
    selection_path = PARENT / 'selected_hyperparameters.json'
    protocol = read_json(protocol_path)
    selection = read_json(selection_path)
    if sha256(PARENT / 'protocol.json') != protocol['upstream_dense_protocol']['sha256']:
        raise RuntimeError('Original parent protocol changed')
    if selection['protocol_sha256'] != protocol['upstream_dense_protocol']['sha256']:
        raise RuntimeError('Parent source-only selection lineage changed')
    bindings = [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)}
                for p in [protocol_path, selection_path, PARENT / 'protocol.json']]
    records = []
    chosen = []
    calibration = [(p, read_json(p)) for p in sorted((PARENT / 'calibration').glob('*/run.json'))]
    for spec in selection['selected']:
        candidates = [(p, run) for p, run in calibration
                      if all(run['model'].get(k) == v for k, v in spec.items() if k != 'name')]
        if sorted(run['seed'] for _, run in candidates) != [11, 22, 33]:
            raise RuntimeError('Exactly three original selected calibration fits required')
        if any(run['study_id'] != read_json(PARENT / 'protocol.json')['study_id'] for _, run in candidates):
            raise RuntimeError('Selected parent study identity changed')
        chosen += [(spec['name'], p, run) for p, run in candidates]
    for spec in protocol['systems']:
        for seed in protocol['seeds']:
            p = SOURCE / 'runs' / (spec['name']+'_seed'+str(seed)) / 'run.json'
            run = read_json(p)
            if run['study_id'] != protocol['study_id']:
                raise RuntimeError('Reference control study identity changed')
            chosen.append((spec['name'], p, run))
    if len(chosen) != 21:
        raise RuntimeError('Four selected parent controls and all three reference variants required')
    annotation_identity = None
    for family, path, run in chosen:
        if sorted(run['training_ids']) != sorted(protocol['source_training_ids']):
            raise RuntimeError('Source training identities differ')
        if run['validation_ids'] != protocol['source_validation_ids']:
            raise RuntimeError('Source validation ordering differs')
        if (run['training_images'], run['validation_images']) != (430, 144):
            raise RuntimeError('Unexpected source supervision')
        reload = run['checkpoint_reload']
        if not reload['fresh_model'] or max(reload['logit_max_abs_error'],
                                           reload['sampled_pixel_probability_max_abs_error']) > 2e-5:
            raise RuntimeError('Original fresh-reload verification failed')
        mask = run['validation_localization']
        if (mask['negative_images_with_structural_annotation'] != 0 or
                mask['pooled_negative_structure_false_positive_fraction'] is not None):
            raise RuntimeError('Recheck the claimed absence of UCF normal-structure supervision')
        identity = [(p['id'], p['known_pixels'], p['crack_pixels']) for p in mask['per_image']]
        if annotation_identity is None:
            annotation_identity = identity
        elif identity != annotation_identity:
            raise RuntimeError('Native annotation support differs across controls')
        metric = run['validation_metrics']
        bindings.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
        records.append({'family': family, 'seed': run['seed'],
                        'image_AUROC': metric['auc'],
                        'native_pixel_AP': mask['pooled_pixel_average_precision'],
                        'positive_image_Dice': mask['mean_positive_image_dice'],
                        'best_round': run['best_round'], 'completed_rounds': run['completed_rounds'],
                        'trainable_parameters': run['trainable_parameters'],
                        'source_logit_threshold': run['source_logit_threshold'],
                        'source_run': str(path.relative_to(ROOT)),
                        'checkpoint_sha256': run['checkpoint_sha256'],
                        'original_fresh_reload_verified': True})
    identity = {'bindings': bindings, 'builder_sha256': sha256(__file__),
                'scope': 'All21 selected original UCF SOURCE-validation fits; descriptive mean and sample SD only',
                'training_images': 430, 'validation_images': 144,
                'validation_mask_support_sha256': hashlib.sha256(json.dumps(annotation_identity).encode()).hexdigest()}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    folder = OUT / key
    if (folder / 'results.json').exists():
        result = read_json(folder / 'results.json')
        if result['identity'] != identity or any(sha256(ROOT / a['path']) != a['sha256'] for a in result['figures']):
            raise RuntimeError('Prior source readout changed')
        print('Reused complete UCF source readout', folder)
        return
    folder.mkdir(parents=True, exist_ok=False)
    summary = []
    for family in LABELS:
        group = sorted((r for r in records if r['family'] == family), key=lambda r: r['seed'])
        if [r['seed'] for r in group] != [11, 22, 33]:
            raise RuntimeError('No partial family means')
        summary.append({'family': family, 'metrics': {metric: {
            'mean': statistics.mean(r[metric] for r in group),
            'sample_SD': statistics.stdev(r[metric] for r in group)} for metric, _ in METRICS}})
    style = dict(plt.rcParamsDefault)
    style.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
                  'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.8), sharey=True)
        fig.subplots_adjust(left=.27, right=.98, bottom=.3, top=.75, wspace=.25)
        for ax, (metric, title) in zip(axes, METRICS):
            for i, row in enumerate(summary):
                color = '#ad7b35' if row['family'].startswith('ucf_reference') else '#426e85'
                point = row['metrics'][metric]
                values = sorted((r for r in records if r['family'] == row['family']), key=lambda r: r['seed'])
                ax.errorbar(point['mean'], i, xerr=point['sample_SD'], fmt='D', ms=3.6,
                            color=color, capsize=2.3, lw=1.1, zorder=3)
                for offset, marker, run in zip([-.08, 0., .08], ['o', '^', 's'], values):
                    ax.scatter(run[metric], i+offset, marker=marker, s=12, color=color, alpha=.4, zorder=2)
            bounds=[r[metric] for r in records]
            bounds += [v for row in summary for v in [row['metrics'][metric]['mean']-row['metrics'][metric]['sample_SD'], row['metrics'][metric]['mean']+row['metrics'][metric]['sample_SD']]]
            ax.set_xlim(max(0.,np.floor((min(bounds)-.03)*20)/20),min(1.,np.ceil((max(bounds)+.03)*20)/20))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3))
            ax.set_title(title, loc='left', pad=12)
            ax.set_yticks(range(len(summary))); ax.set_yticklabels(LABELS.values()); ax.set_ylim(len(summary)-.5, -.6)
            ax.tick_params(axis='y', length=0, pad=8); ax.grid(axis='x', color='#e8edef', lw=.7)
            ax.axhline(3.5, color='#dfe6e8', lw=.8)
            for side in ['top', 'right', 'left']: ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color('#9aabb1')
        fig.text(.27, .93, 'UCF source controls: all original fits', fontsize=12, weight='bold', color='#183f4d')
        fig.text(.27, .86, '430 training masks + 144 reused validation masks', fontsize=8.8)
        notes = ['Diamonds and bars: mean ± sample SD; faint markers: all three seeds (11 circle, 22 triangle, 33 square).',
                 'Blue: selected original parent controls; gold: all 9 additional reference-control fits. Higher is better.',
                 'Axes use the displayed metric-specific ranges.',
                 'The same validation labels select checkpoints and parent hyperparameters; these are optimistic source summaries.',
                 'Native AP and positive-image Dice are different endpoints. No independent comparison or confidence interval.',
                 'UCF real/simulated provenance is unresolved per file. This readout adds no target scores or model inference.']
        for i, note in enumerate(notes): fig.text(.06, .225-i*.033, note, fontsize=6.75, color='#536d78')
        assets = []
        for ext in ['png', 'pdf', 'svg']:
            p = folder / ('source_controls.'+ext); fig.savefig(p, dpi=240, facecolor='white')
            assets.append({'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)})
        plt.close(fig)
    csv_path = folder / 'all21_source_runs.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    result = {'at': now(), 'identity': identity, 'source_runs': records, 'summary': summary,
              'figures': assets, 'csv': {'path': str(csv_path.relative_to(ROOT)), 'sha256': sha256(csv_path)},
              'normal_structure_pixel_FPR': 'Not evaluated: UCF validation has no annotated normal-structure support',
              'new_training_or_inference': False, 'final_confirmation_access': False}
    save_json(folder / 'results.json', result)
    save_json(OUT / 'latest.json', {'path': str((folder / 'results.json').relative_to(ROOT)), 'sha256': sha256(folder / 'results.json')})
    print('Rendered complete21-fit UCF source comparison', folder)


if __name__ == '__main__':
    main()
