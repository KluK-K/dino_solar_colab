"""Render verified fixed held-source maps, with a separate source-only layout check.

No model construction/inference or score-based case selection. The layout
check never represents held-source maps and is excluded from research results.
"""
import argparse
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from build_dense_calibration_report import outline, roi
from elresearch.common import ROOT, read_json, save_json, sha256, digest, now
from elresearch.data import IGNORE, load_image, load_mask
from elresearch.publication_splits import fold_records

BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'dense_development_v1'
REPLAY = BASE / 'dense_held_source_map_replay_v1'
OUT = BASE / 'dense_held_native_gallery_v1'
ARMS = [('deeplabv3plus_full', 'Adapted CNN'), ('dinov3_dense_frozen', 'DINO frozen'),
        ('dinov3_dense_lora', 'DINO LoRA'), ('dinov3_dense_reference', 'Normal reference')]


def load_maps(items, by):
    maps = {}; bindings = []
    for item in items:
        identifier, path = item['id'], ROOT / item['path']
        if identifier not in by or sha256(path) != item['sha256']:
            raise RuntimeError('Fixed native map identity/payload changed')
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['id']) != identifier: raise RuntimeError('Fixed map contains another image')
            value = saved['score'].astype(np.float32)
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise RuntimeError('Use original finite softmax maps on fixed0..1 scale')
        maps[identifier] = value; bindings.append({'path': item['path'], 'sha256': item['sha256']})
    if len(maps) != 8: raise RuntimeError('Retain all8 original fixed native cases')
    return maps, bindings


def render(fold, budget, models, by, bindings, mode):
    ids = set(models[0]['maps'])
    if len(models) != 4 or any(set(m['maps']) != ids for m in models):
        raise RuntimeError('Same8 cases must be available for all4 systems')
    native = {}; gt_bindings = []
    for identifier in ids:
        row = by[identifier]
        if row['partition'] not in {'validation', 'target_development'}:
            raise PermissionError('Only original source/held development cases are allowed')
        image, mask = np.asarray(load_image(row)), load_mask(row)
        if image.shape != mask.shape or any(m['maps'][identifier].shape != mask.shape for m in models):
            raise RuntimeError('Native EL, GT and every model map must align')
        known, crack = int((mask != IGNORE).sum()), int((mask == 1).sum())
        for model in models:
            s = model['stats'][identifier]
            if (known, crack) != (s['known_pixels'], s['crack_pixels']):
                raise RuntimeError('Original native annotations do not match displayed truth')
        native[identifier] = (image, mask)
        gt_bindings.append({'id': identifier, 'image_sha256': row['image_sha256'], 'mask_sha256': row['mask_sha256']})
    negatives = sorted(i for i in ids if not by[i]['target'])
    positives = sorted((i for i in ids if by[i]['target']),
                       key=lambda i: (models[0]['stats'][i]['annotated_crack_fraction'], i))
    if len(negatives) != 3 or len(positives) != 5:
        raise RuntimeError('Preserve original3 identity negatives and5 annotation-quantile positives')
    all_ids = negatives + positives; main_ids = [negatives[0], positives[0], positives[-1]]
    identity = {'builder_sha256': sha256(__file__), 'roi_outline_sha256': sha256(ROOT / 'build_dense_calibration_report.py'),
                'canonical_label_loader_sha256': sha256(ROOT / 'elresearch/data.py'),
                'source_split_loader_sha256': sha256(ROOT / 'elresearch/publication_splits.py'),
                'source_protocol_sha256': sha256(SOURCE / 'protocol.json'),
                'fold': fold, 'budget': budget, 'mode': mode, 'bindings': bindings,
                'native_annotation_bindings': sorted(gt_bindings, key=lambda r: r['id']),
                'all_ids': all_ids, 'main_ids': main_ids,
                'display_rule': 'Main uses same GT/context ROI and global0.65 heatmap alpha over native EL. Supplement retains every case as raw linear0..1 softmax maps. No per-image normalization.'}
    directory = OUT / ('source_validation_layout_only' if mode == 'source_validation_layout_only' else 'held_source') / digest(identity)[:20]
    directory.mkdir(parents=True, exist_ok=True); result_path = directory / 'figure_inputs.json'
    if result_path.exists():
        previous = read_json(result_path)
        if previous['identity'] != identity: raise RuntimeError('Preserve completed native illustration snapshot')
        for f in previous['figures']:
            if sha256(ROOT / f['path']) != f['sha256']: raise RuntimeError('Completed native figure changed')
        return previous
    cmap = plt.get_cmap('magma').copy(); cmap.set_bad('#ccd4d8')
    gray = plt.get_cmap('gray').copy(); gray.set_bad('#ccd4d8')
    labelmap = ListedColormap(['#20292f', '#36dbd0', '#e9993c', '#ccd4d8'])
    style = {'font.family': 'DejaVu Sans', 'font.size': 7., 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
    cohort = fold.removeprefix('holdout_')
    for selected, zoom, stem in [(main_ids, True, 'shared_rois_EL_overlay'), (all_ids, False, 'all8_raw_probability_maps')]:
        height = 4.8 if zoom else 9.4
        with plt.rc_context(matplotlib.rcParamsDefault), plt.rc_context(style):
            fig, axes = plt.subplots(len(selected), 6, figsize=(7.2, height), squeeze=False)
            for r, identifier in enumerate(selected):
                image, mask = native[identifier]; left, top, right, bottom = roi(mask)
                crop = (slice(top, bottom), slice(left, right)) if zoom else (slice(None), slice(None))
                axes[r, 0].imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if zoom:
                    axes[r, 0].add_patch(plt.Rectangle((left, top), right-left, bottom-top, fill=False, color='white', lw=.65))
                    axes[r, 1].imshow(image[crop], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    outline(axes[r, 1], mask[crop])
                else:
                    labels = np.where(mask == IGNORE, 3, mask)
                    axes[r, 1].imshow(labels, cmap=labelmap, norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4), interpolation='nearest')
                parts = identifier.rsplit('/', 1)[-1].removesuffix('.png').split('_', 1)
                axes[r, 0].set_ylabel('\n'.join(parts), fontsize=5.8)
                for col, model in enumerate(models, 2):
                    score = model['maps'][identifier][crop]; truth = mask[crop]
                    if zoom:
                        axes[r, col].imshow(np.ma.masked_where(truth == IGNORE, image[crop]), cmap=gray,
                                            vmin=0, vmax=255, interpolation='nearest')
                    axes[r, col].imshow(np.ma.masked_where(truth == IGNORE, score), cmap=cmap,
                                        vmin=0, vmax=1, interpolation='nearest', alpha=.65 if zoom else 1.)
                    outline(axes[r, col], truth, score)
                for ax in axes[r]:
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values(): spine.set_visible(False)
            labels = ['Native EL', 'EL + GT\n(shared ROI)' if zoom else 'Canonical\nannotation'] + [label for _, label in ARMS]
            for ax, label in zip(axes[0], labels): ax.set_title(label, fontsize=6.7, pad=6)
            fig.subplots_adjust(left=.08, right=.985, top=1-.68/height, bottom=1.0/height, wspace=.10, hspace=.18)
            title = ('SOURCE VALIDATION LAYOUT CHECK · not held-source predictions' if mode == 'source_validation_layout_only'
                     else f'Held {cohort} · {100*budget:g}% source training masks · fixed seed11 cases')
            fig.suptitle(title, x=.08, ha='left', fontsize=8.5, y=.98, color='#244a56')
            cax = fig.add_axes([.62, .70/height, .31, .075/height])
            fig.colorbar(ScalarMappable(norm=plt.Normalize(0,1), cmap=cmap), cax=cax, orientation='horizontal', ticks=[0,.5,1])
            cax.tick_params(labelsize=6, length=2, pad=1)
            fig.text(.62,.84/height,'Crack softmax score [0,1]',fontsize=6.3,color='#536f79')
            fig.text(.08,.70/height,'Cyan: crack GT; white: displayed score >0.5; gray: ignored.',fontsize=6.3,color='#536f79')
            if not zoom: fig.text(.08,.54/height,'Orange: annotated other defect.',fontsize=6.3,color='#536f79')
            fig.text(.08,.42/height,('Shared ROI; heatmap opacity 0.65 over native EL. Raw maps retained for all 8 cases.' if zoom else
                                   'All 8 original fixed cases; raw linear [0,1] scale, identical across every model.'),fontsize=6.3,color='#536f79')
            fig.text(.08,.23/height,'Saved float16 maps illustrate seed 11; full-float evaluation supplies the three-seed metrics.',fontsize=6.3,color='#536f79')
            fig.text(.08,.06/height,('Layout check uses source validation only; not a new held-source result.' if mode == 'source_validation_layout_only' else
                                   'Same cases/GT-defined views for all4 systems; development evidence, not final confirmation.'),fontsize=6.3,color='#536f79')
            for ext in ['png','pdf','svg']: fig.savefig(directory / f'{stem}.{ext}', dpi=300, facecolor='white')
            plt.close(fig)
    result = {'at': now(), 'identity': identity, 'research_held_source_figure': mode != 'source_validation_layout_only',
              'new_model_inference': False, 'figures': [{'path': str((directory / f'{stem}.{ext}').relative_to(ROOT)), 'sha256': sha256(directory / f'{stem}.{ext}')}
              for stem in ['shared_rois_EL_overlay','all8_raw_probability_maps'] for ext in ['png','pdf','svg']]}
    save_json(result_path, result); return result


def source_layout_check():
    settings = read_json(SOURCE / 'protocol.json'); _, source, _ = fold_records(settings['split_manifest'], 'holdout_ARTS')
    by = {r['id']: r for r in source}; models = []; bindings = []
    for family, _ in ARMS:
        folder = SOURCE / 'runs/holdout_ARTS/budget_0.2' / f'{family}_seed11'
        run = read_json(folder / 'run.json'); bindings.append({'path': str((folder / 'run.json').relative_to(ROOT)), 'sha256': sha256(folder / 'run.json')})
        items = []
        for identifier in run['validation_localization']['fixed_example_ids']:
            path = folder / 'fixed_validation_examples' / (digest(identifier) + '.npz')
            items.append({'id': identifier, 'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
        maps, data = load_maps(items, by); bindings += data
        models.append({'family': family, 'maps': maps, 'stats': {r['id']: r for r in run['validation_localization']['per_image']}})
    result = render('holdout_ARTS', .2, models, by, bindings, 'source_validation_layout_only')
    save_json(OUT / 'layout_check_latest.json', {'at': now(), 'figure_inputs': result, 'research_held_source_result': False})
    print('Source-validation-only layout check:', result['figures'][0]['path'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--layout-check', action='store_true'); args = parser.parse_args()
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Source development figures must precede final confirmation scoring')
    OUT.mkdir(exist_ok=True)
    if args.layout_check: source_layout_check(); return
    state = read_json(REPLAY / 'status.json')
    if state['state'] != 'complete' or state['completed_jobs'] != 24:
        print('Verified held-source native exports pending; no held-source figure created.'); return
    summary = read_json(REPLAY / 'summary.json'); plan = read_json(REPLAY / 'plan.json')
    if summary['plan_sha256'] != sha256(REPLAY / 'plan.json') or plan['source_protocol_sha256'] != sha256(SOURCE / 'protocol.json'):
        raise RuntimeError('Original native-replay source changed')
    if len(summary['source_receipts']) != 24:
        raise RuntimeError('Require all24 original replay receipts')
    cells = [c for c in read_json(BASE / 'dense_development_complete_cells.json')['complete_cells'] if c['fold'] != 'internal']
    if len(cells) != 6: raise RuntimeError('Refresh every completed dense source cell before rendering')
    groups = {}
    for item in summary['source_receipts']:
        receipt_path = ROOT / item['path']
        if sha256(receipt_path) != item['sha256']: raise RuntimeError('Verified map receipt changed')
        r = read_json(receipt_path)
        if r['plan_sha256'] != summary['plan_sha256']: raise RuntimeError('Map export uses a different plan')
        if r['job'] not in plan['jobs'] or r['job']['seed'] != 11:
            raise RuntimeError('Native receipt uses an undeclared model/source/seed')
        entries=groups.setdefault((r['job']['fold'], r['job']['budget']), {})
        if r['job']['family'] in entries:raise RuntimeError('Duplicate model receipt in native gallery')
        entries[r['job']['family']] = (r, item)
    if len(groups) != 6: raise RuntimeError('Retain all6 held-source cells')
    settings = read_json(SOURCE / 'protocol.json'); pages = []
    for (fold, budget), entries in sorted(groups.items()):
        if set(entries) != {a[0] for a in ARMS}: raise RuntimeError('Keep all4 matched source systems')
        _, _, target = fold_records(settings['split_manifest'], fold); by = {r['id']: r for r in target}
        models = []; bindings = []
        for family, _ in ARMS:
            receipt, item = entries[family]; bindings.append(item)
            for binding in receipt['source_bindings']:
                if sha256(ROOT / binding['path']) != binding['sha256']: raise RuntimeError('Original fitted source changed')
            evaluation_path = next(b['path'] for b in receipt['source_bindings'] if b['path'].endswith('/target_development.json'))
            evaluation = read_json(ROOT / evaluation_path)
            if {m['id'] for m in receipt['maps']} != set(evaluation['localization']['fixed_example_ids']):
                raise RuntimeError('Preserve original target case IDs without selection')
            maps, data = load_maps(receipt['maps'], by); bindings += data
            models.append({'family': family, 'maps': maps, 'stats': {r['id']: r for r in evaluation['localization']['per_image']}})
        result = render(fold, budget, models, by, bindings, 'verified_held_source')
        pages.append({'fold': fold, 'budget': budget, 'figure_inputs': result})
    save_json(OUT / 'index.json', {'at': now(), 'pages': pages, 'complete_cells': 6, 'native_maps': 192,
                                 'scope': 'Verified seed11 maps; original all3-seed quantitative results stay separate. No new model or threshold.'})
    body='<!doctype html><html lang="en"><meta charset="utf-8"><title>Verified held-source native comparisons</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#244a56}img{width:100%}section{border-top:1px solid #dbe5e8;margin-top:30px;padding-top:16px}</style><h1>Verified native held-source comparisons</h1><p>All six source-held-out development cells, four systems and original eight fixed cases. Main panels blend fixed0–1 heatmaps over native EL with global opacity0.65; supplements preserve all raw maps. Seed11 illustrations are separate from original three-seed metrics.</p>'
    for p in pages:
        images=p['figure_inputs']['figures'];main=str((ROOT/images[0]['path']).relative_to(OUT));full=str((ROOT/images[3]['path']).relative_to(OUT))
        body+=f'<section><h2>{html.escape(p["fold"])} · {100*p["budget"]:g}% source masks</h2><img src="{main}"><p><a href="{main.replace(".png",".pdf")}">Vector main figure</a></p><details><summary>All8 fixed cases; raw probability maps</summary><img src="{full}"></details></section>'
    (OUT / 'report.html').write_text(body+'</html>'); print('Verified held-source galleries:6 cells,192 original fixed-case maps')


if __name__ == '__main__': main()
