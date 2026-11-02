"""Matched saved-map figures for complete real-SCDD source-validation cells.

No inference, per-image color normalization, new case search or confirmation
access. The original eight fixed examples and all completed model families are
retained. Quantitative annotations come from full-float original evaluations.
"""
import csv
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from build_dense_calibration_report import outline, roi
from build_dense_development_progress import NAMES
from elresearch.common import ROOT, digest, now, read_json, save_json, sha256
from elresearch.data import IGNORE, load_image, load_mask, records

BASE = ROOT / 'artifacts/publication_20261017'
OUTPUT = BASE / 'dense_real_source_gallery_v1'


def main():
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Source illustrations must precede real confirmation scoring')
    cells = read_json(BASE / 'dense_development_complete_cells.json')['complete_cells']
    by = {r['id']: r for r in records('scdd_publication_v1', 'validation')}
    OUTPUT.mkdir(exist_ok=True)
    pages = []
    for cell in cells:
        if cell['fold'] != 'internal':
            continue
        models = []; bindings = []; original_ids = None; numeric = []
        for model in cell['models']:
            fit = next(f for f in model['fits'] if f['seed'] == 11)
            path = ROOT / fit['run']; run = read_json(path)
            if sha256(path) != fit['run_sha256']:
                raise RuntimeError('Completed source fit changed')
            ids = run['validation_localization']['fixed_example_ids']
            if original_ids is None:
                original_ids = ids
            if ids != original_ids or not set(ids) <= set(by):
                raise RuntimeError('Original gallery source identities must match every model')
            stats = {r['id']: r for r in run['validation_localization']['per_image']}
            with np.load(ROOT / fit['predictions'], allow_pickle=False) as predictions:
                logits = dict(zip(predictions['ids'].tolist(), predictions['logits'].tolist()))
            maps = {}
            for identifier in ids:
                source = path.parent / 'fixed_validation_examples' / (digest(identifier) + '.npz')
                with np.load(source, allow_pickle=False) as saved:
                    if str(saved['id']) != identifier:
                        raise RuntimeError('Saved native map identity mismatch')
                    score = saved['score'].astype(np.float32)
                if not np.isfinite(score).all() or score.min() < 0 or score.max() > 1:
                    raise RuntimeError('Softmax display values must be finite in [0,1]')
                maps[identifier] = score
                bindings.append({'map': str(source.relative_to(ROOT)), 'sha256': sha256(source)})
                numeric.append({'id': identifier, 'model': model['family'], 'seed': 11,
                    'source_logit_threshold': run['source_logit_threshold'],
                    'image_logit': logits[identifier],
                    'image_detected': logits[identifier] > run['source_logit_threshold'],
                    'known_pixel_crack_fraction': stats[identifier]['annotated_crack_fraction'],
                    'positive_image_Dice': stats[identifier]['positive_image_dice'],
                    'known_noncrack_pixel_FPR': stats[identifier]['noncrack_pixel_false_positive_fraction']})
            models.append({'family': model['family'], 'maps': maps, 'stats': stats})
        negatives = sorted(i for i in original_ids if not by[i]['target'])
        positives = sorted((i for i in original_ids if by[i]['target']),
            key=lambda i: (models[0]['stats'][i]['annotated_crack_fraction'], i))
        if not negatives or len(positives) < 2 or len(original_ids) != 8:
            raise RuntimeError('Keep the declared eight-case source gallery')
        all_ids = negatives + positives
        main_ids = [negatives[0], positives[0], positives[-1]]
        identity = {'cell_sha256': cell['cell_sha256'], 'generator_sha256': sha256(__file__),
            'roi_and_outline_helper_sha256': sha256(ROOT / 'build_dense_calibration_report.py'),
            'map_bindings': bindings}
        directory = OUTPUT / digest(identity)[:20]
        directory.mkdir(exist_ok=True)
        plan = {'identity': identity, 'cell_id': cell['cell_id'], 'budget': cell['budget'],
            'seed': 11, 'all_ids': all_ids, 'main_ids': main_ids,
            'case_selection': 'Retrospective source illustration. All8 original fixed cases retained. Main panels use first negative and smallest/largest canonical known-pixel crack area among the fixed positives; no model score selects cases.',
            'roi': 'Same GT bounding box plus30% context, minimum96 native pixels, clipped to image; negative centered45%-shorter-edge view. Raw full-field context and complete gallery retained.',
            'scales': 'EL0–255, softmax0–1 fixed across every model/image. Cyan GT, white dashed score>0.5, ignored annotation gray. No per-image normalization.',
            'render_style': 'Explicit Matplotlib default rc_context with DejaVu Sans9 and embedded vector fonts; caller plotting state cannot alter this figure.',
            'limits': 'Seed11 saved float16 maps are illustrative. Numbers use original full-float native evaluation. Source validation selected checkpoints; not independent test or a semi-supervised paper reproduction.'}
        if (directory / 'plan.json').exists():
            if read_json(directory / 'plan.json') != plan:
                raise RuntimeError('Preserve completed visualization specification')
        else:
            save_json(directory / 'plan.json', plan)
        native = {i: (np.asarray(load_image(by[i]).convert('L')), load_mask(by[i])) for i in all_ids}
        for identifier, (image, mask) in native.items():
            if image.shape != mask.shape or any(m['maps'][identifier].shape != mask.shape for m in models):
                raise RuntimeError('Native field, canonical truth and all model maps must align')
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
        cmap = plt.get_cmap('magma').copy(); cmap.set_bad('#ccd4d8')
        width = max(12.8, 1.7 * (2 + len(models)))

        def gallery(identifiers, zoom, name):
            with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
                plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
                gallery_impl(identifiers, zoom, name)

        def gallery_impl(identifiers, zoom, name):
            height = 2.15 * len(identifiers) + 1.9
            fig, axes = plt.subplots(len(identifiers), 2 + len(models), figsize=(width, height), squeeze=False)
            for row, identifier in enumerate(identifiers):
                image, mask = native[identifier]; left, top, right, bottom = roi(mask)
                crop = (slice(top, bottom), slice(left, right))
                axes[row, 0].imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if zoom:
                    axes[row, 0].add_patch(plt.Rectangle((left, top), right-left, bottom-top, fill=False, edgecolor='white', lw=.9))
                    axes[row, 1].imshow(image[crop], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    outline(axes[row, 1], mask[crop])
                else:
                    labels = np.where(mask == IGNORE, 3, mask)
                    axes[row, 1].imshow(labels, cmap=ListedColormap(['#20292f', '#36dbd0', '#e9993c', '#ccd4d8']),
                        norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), interpolation='nearest')
                name_parts = identifier.split('/')[-1].removesuffix('.png').split('_', 1)
                axes[row, 0].set_ylabel('\n'.join(name_parts), fontsize=7.3)
                for column, model in enumerate(models, 2):
                    score = model['maps'][identifier][crop] if zoom else model['maps'][identifier]
                    shown_mask = mask[crop] if zoom else mask
                    axes[row, column].imshow(np.ma.masked_where(shown_mask == IGNORE, score),
                        cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                    outline(axes[row, column], shown_mask, score)
                for ax in axes[row]:
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values(): spine.set_visible(False)
            labels = ['Native EL', 'EL + crack truth\n(shared ROI)' if zoom else 'Canonical\nannotation']
            labels += [NAMES[m['family']].replace(' / ', '\n').replace(' + ', '\n+ ').replace('DeepLabv3+', 'DeepLabv3+') for m in models]
            for ax, label in zip(axes[0], labels): ax.set_title(label, fontsize=8.1, pad=9)
            fig.subplots_adjust(left=.075, right=.99, top=1-1.0/height, bottom=1.5/height, wspace=.08, hspace=.15)
            fig.suptitle(f"Real SCDD · {int(cell['budget']*100)}% train masks · fixed source-validation cases · seed11",
                x=.075, ha='left', fontsize=12.5, y=.985, color='#193e48')
            cax = fig.add_axes([.59, .89/height, .36, .12/height])
            fig.colorbar(ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap), cax=cax,
                orientation='horizontal', ticks=[0, .25, .5, .75, 1]).set_label('Crack softmax score · fixed [0,1]', fontsize=8)
            fig.text(.075, .9/height, 'Cyan: crack truth\nWhite dashed: score >0.5; gray: ignored', fontsize=8, color='#506d77')
            fig.text(.075, .12/height, 'Same images, annotations and views for every model; no outcome-based case selection. Saved float16 maps are illustrative.\n'
                'Source validation selects checkpoints; source-validation masks and UCF calibration are additional label exposure. All eight cases retained in full gallery.',
                fontsize=7.7, color='#506d77', linespacing=1.45)
            for extension in ['png', 'pdf', 'svg']: fig.savefig(directory / f'{name}.{extension}', dpi=220, facecolor='white')
            plt.close(fig)

        gallery(main_ids, True, 'shared_native_rois')
        gallery(all_ids, False, 'all_eight_fixed_cases')
        with (directory / 'fixed_case_metrics.csv').open('w') as output:
            writer = csv.DictWriter(output, fieldnames=list(numeric[0])); writer.writeheader(); writer.writerows(numeric)
        save_json(directory / 'figure_inputs.json', {'plan_sha256': sha256(directory / 'plan.json'),
            'all_seed11_case_metrics': numeric, 'source_cell': cell['cell_path'],
            'all3_seed_aggregate_in_cell': True, 'no_new_inference': True})
        pages.append({'folder': str(directory.relative_to(OUTPUT)), 'cell_id': cell['cell_id'], 'budget': cell['budget']})
    OUTPUT.joinpath('report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Real EL native localization comparison</title><style>body{font:16px/1.6 system-ui;color:#193e48;background:#f3f6f7;margin:0}main{max-width:1260px;margin:35px auto;padding:26px;background:white}img{width:100%}section{padding-top:20px;border-top:1px solid #dce5e7;margin-top:24px}a{color:#216f82}</style><main><h1>Real EL images, manual truth and matched model predictions</h1><p>Only completed real-SCDD cells with every declared system and all three fitted seeds enter this gallery. The original eight fixed cases are retained for each cell. Figures illustrate seed11, with identical GT-defined crops and fixed color scales. Quantitative three-seed results remain in the complete-cell report. These validation images selected checkpoints; they are development evidence.</p>'''
        + ''.join(f'<section><h2>{int(p["budget"]*100)}% training masks</h2><img src="{p["folder"]}/shared_native_rois.png" alt="Identical real EL crops and every completed model"><p><a href="{p["folder"]}/shared_native_rois.pdf">Vector PDF</a> · <a href="{p["folder"]}/fixed_case_metrics.csv">Original numerical case results</a> · <a href="{p["folder"]}/plan.json">Case and display policy</a></p><details><summary>All eight original fixed cases; no omitted failure</summary><img src="{p["folder"]}/all_eight_fixed_cases.png" alt="All eight native source-validation cases with complete model comparison"></details></section>' for p in pages)
        + '</main></html>')
    save_json(OUTPUT / 'index.json', {'at': now(), 'pages': pages, 'scope': 'Retrospective saved-map source-validation illustrations; no new inference or confirmation access.'})
    print('Complete real-SCDD native galleries', len(pages), flush=True)


if __name__ == '__main__':
    main()
