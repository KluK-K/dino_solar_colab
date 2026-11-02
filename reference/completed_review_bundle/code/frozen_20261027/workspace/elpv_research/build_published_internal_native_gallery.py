"""All original eight source cases, including the full published-method ports.

Saved-map illustration only: no network, inference, threshold fit, bootstrap or
new case selection. Every applicable dense system and published inference view
is retained, after their all-model/three-seed internal comparison is complete.
"""
import csv
import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from build_dense_calibration_report import outline, roi
from elresearch.common import ROOT, digest, now, read_json, save_json, sha256
from elresearch.data import IGNORE, load_image, load_mask, records


BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'published_internal_native_gallery_v1'
LABELS = {'deeplabv3plus_full': 'Adapted\nCNN', 'dinov3_dense_frozen': 'DINO\nfrozen',
          'dinov3_dense_lora': 'DINO\nLoRA', 'dinov3_dense_full': 'DINO\nfull',
          'dinov3_dense_reference': 'DINO\nnormal ref.',
          'dinov3_reference_no_position': 'Reference\nno position',
          'dinov3_reference_mean_evidence': 'Reference\nmean evidence',
          'published_deeplab_fsl/primary': 'Full port\nsupervised',
          'published_psmt_ssl/primary': 'Full SSL port\nteacher1',
          'published_psmt_ssl/ensemble': 'Full SSL port\nensemble'}


def guard():
    state = read_json(BASE / 'inference_efficiency_v1/status.json')['state']
    if not (state.startswith('queued') or state == 'complete'):
        raise PermissionError('Do not render during formal efficiency timings')
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Source illustrations must precede final model scoring')


def bind(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def immutable(path, value):
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError('Preserve original gallery registration: ' + str(path))
    else:
        save_json(path, value)


def main():
    guard()
    complete_path = BASE / 'published_complete_internal_readout_v1/4044360fa3f4fee05274/results.json'
    complete = read_json(complete_path)
    if (sha256(complete_path) != '1df338c12e8130770f429f5b97ebe6bb4a2cf51663e1fe4ee0ee828a9ac6c550'
            or complete['unique_completed_fits_reused'] != 51):
        raise RuntimeError('Original all-model/three-seed internal comparison changed')
    comparison_plan_path = ROOT / complete['plan']['path']
    if sha256(comparison_plan_path) != complete['plan']['sha256']:
        raise RuntimeError('Original completed comparison registration changed')
    comparison_bindings = {row['path']: row['sha256']
                           for row in read_json(comparison_plan_path)['input_bindings']}
    dense_index_path = BASE / 'dense_development_complete_cells.json'
    dense = read_json(dense_index_path)
    original_index_path = BASE / 'dense_real_source_gallery_v1/index.json'
    original = read_json(original_index_path)
    by_id = {row['id']: row for row in records('scdd_publication_v1', 'validation')}
    pages = []
    for budget in [.2, 1.]:
        guard()
        cell = next(c for c in dense['complete_cells'] if c['fold'] == 'internal' and c['budget'] == budget)
        if sha256(ROOT / cell['cell_path']) != cell['cell_sha256']:
            raise RuntimeError('Original complete dense cell changed')
        original_page = next(page for page in original['pages'] if page['budget'] == budget)
        original_plan_path = original_index_path.parent / original_page['folder'] / 'plan.json'
        original_plan = read_json(original_plan_path)
        ids = original_plan['all_ids']
        if len(ids) != 8 or len(set(ids)) != 8 or not set(ids) <= set(by_id):
            raise RuntimeError('Keep exactly every original fixed source case')
        for row in original_plan['identity']['map_bindings']:
            if sha256(ROOT / row['map']) != row['sha256']:
                raise RuntimeError('An original dense gallery map changed')
        inputs = [Path(__file__).absolute(), complete_path, comparison_plan_path, dense_index_path,
                  ROOT / cell['cell_path'], original_index_path, original_plan_path,
                  ROOT / 'build_dense_calibration_report.py', ROOT / 'elresearch/data.py',
                  ROOT / 'data/scdd_publication_v1/manifest.json']
        jobs = []
        for model in cell['models']:
            if sorted(fit['seed'] for fit in model['fits']) != [11, 22, 33]:
                raise RuntimeError('All three dense fitted seeds are required')
            fit = next(fit for fit in model['fits'] if fit['seed'] == 11)
            run_path = ROOT / fit['run']
            if sha256(run_path) != fit['run_sha256']:
                raise RuntimeError('Original dense source fit changed')
            jobs.append((model['family'], run_path, 'primary'))
        ports = [('published_deeplab_fsl', ['primary'])]
        if budget == .2:
            ports.append(('published_psmt_ssl', ['primary', 'ensemble']))
        for name, views in ports:
            for seed in [11, 22, 33]:
                path = BASE / 'published_methods_v1/runs/internal' / f'budget_{budget:g}' / f'{name}_seed{seed}/run.json'
                if not path.exists() or (path.parent / 'latest.pt').exists():
                    raise RuntimeError('Complete every original port seed before illustration')
                if sha256(path) != comparison_bindings[str(path.relative_to(ROOT))]:
                    raise RuntimeError('Completed three-seed port input changed')
                inputs.append(path)
            path = BASE / 'published_methods_v1/runs/internal' / f'budget_{budget:g}' / f'{name}_seed11/run.json'
            jobs += [(name + '/' + view, path, view) for view in views]
        models, numeric = [], []
        for system, path, view in jobs:
            if sha256(path) != comparison_bindings[str(path.relative_to(ROOT))]:
                raise RuntimeError('Original source comparison input changed')
            run = read_json(path)
            evaluated = run['inference_views'][view] if 'inference_views' in run else run
            if set(evaluated['validation_localization']['fixed_example_ids']) != set(ids):
                raise RuntimeError('Every model/view must retain the same original eight identities')
            inputs.append(path)
            stats = {row['id']: row for row in evaluated['validation_localization']['per_image']}
            folder = path.parent if view == 'primary' else path.parent / 'inference_views' / view
            maps = {}
            for identifier in ids:
                source = folder / 'fixed_validation_examples' / (digest(identifier) + '.npz')
                inputs.append(source)
                with np.load(source, allow_pickle=False) as saved:
                    if str(saved['id']) != identifier:
                        raise RuntimeError('Saved display-map identity differs')
                    score = saved['score'].astype(np.float32)
                if not np.isfinite(score).all() or score.min() < 0 or score.max() > 1:
                    raise RuntimeError('Invalid softmax display map')
                maps[identifier] = score
                row = stats[identifier]
                numeric.append({'id': identifier, 'system': system, 'seed': 11,
                                'training_masks': run['training_images'], 'validation_masks': run['validation_images'],
                                'image_logit_threshold': evaluated['source_logit_threshold'],
                                'original_full_float_positive_Dice': row['positive_image_dice'],
                                'original_full_float_predicted_pixels': row['predicted_crack_pixels'],
                                'original_full_float_false_positive_pixels': row['false_positive_pixels']})
            models.append({'system': system, 'maps': maps, 'stats': stats,
                           'training_masks': run['training_images'], 'validation_masks': run['validation_images']})
        if len(models) != (10 if budget == .2 else 8):
            raise RuntimeError('Every declared original dense family and port view is required')
        native = {}
        for identifier in ids:
            row = by_id[identifier]
            for key, expected in [('image_path', 'image_sha256'), ('mask_path', 'mask_sha256')]:
                path = ROOT / row[key]
                if sha256(path) != row[expected]:
                    raise RuntimeError('Original source pixels or annotation changed')
                inputs.append(path)
            image, mask = np.asarray(load_image(row).convert('L')), load_mask(row)
            if image.shape != mask.shape or any(model['maps'][identifier].shape != mask.shape for model in models):
                raise RuntimeError('Align native fields, canonical truth and all saved maps')
            native[identifier] = image, mask
        plan = {'version': 1, 'kind': 'Retrospective complete internal source saved-map illustration',
                'budget': budget, 'fixed_seed': 11, 'original_all_eight_ids': ids,
                'systems': [model['system'] for model in models], 'input_bindings': [bind(path) for path in sorted(set(inputs))],
                'case_policy': 'Exactly the original eight cases, in their original gallery order. No new case search or outcome ranking. The same eight source images are reused at both budgets.',
                'display': 'EL0..255 and crack softmax0..1, common fixed scales. Cyan ground truth; white dashed score>0.5. Full native fields and GT-defined shared ROIs are both retained.',
                'roi_policy': original_plan['roi'], 'maps': 'Original saved float16 display arrays only. Contours can differ near0.5 from original full-float quantitative masks; never recompute metrics from display arrays.',
                'supervision': 'Training masks plus120 source-validation masks; upstream UCF574 label/HPO exposure additional. SSL additionally uses remaining source-training images.',
                'comparison_scope': 'All-model/three-seed internal comparison already complete. Figures show the fixed seed11 only; teacher and ensemble share one fitted checkpoint.',
                'new_model_inference_fit_or_bootstrap': False, 'independent_confirmation': False,
                'limits': 'Observed validation images selected checkpoints. Local full-architecture ports are adaptations, not reproduced author-paper scores. Pretraining, losses and scoring differ. These figures and raw pixels are not authorized private-handoff exports.'}
        directory = OUT / digest(plan)[:20]
        immutable(directory / 'plan.json', plan)
        if (directory / 'outputs.json').exists():
            for value in read_json(directory / 'outputs.json')['files']:
                if sha256(ROOT / value['path']) != value['sha256']:
                    raise RuntimeError('Previously completed gallery changed')
        else:
            render(directory, budget, ids, native, models)
            with (directory / 'fixed_case_metrics.csv').open('x', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(numeric[0])); writer.writeheader(); writer.writerows(numeric)
            save_json(directory / 'figure_inputs.json', {'plan': bind(directory / 'plan.json'),
                      'unique_source_images': 8, 'illustrated_seed': 11, 'case_model_records': numeric,
                      'new_inference_or_pixel_metric_recomputation': False})
            files = [p for p in directory.iterdir() if p.is_file()]
            save_json(directory / 'outputs.json', {'at': now(), 'files': [bind(path) for path in sorted(files)],
                      'visual_QA_pending': True, 'manuscript_integrated': False})
        pages.append({'budget': budget, 'folder': str(directory.relative_to(OUT)),
                      'models_and_views': len(models), 'fixed_cases': 8,
                      'plan': bind(directory / 'plan.json'), 'outputs': bind(directory / 'outputs.json')})
    report = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Full-method source native gallery</title><style>body{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:20px;color:#193e48}img{width:100%}section{border-top:1px solid #ddd;margin-top:30px}a{color:#216f82}</style><h1>All eight original source cases, including full-method ports</h1><p>Same eight source-validation images at two budgets; fixed seed11, all applicable systems/views. Complete three-seed numerical comparisons are separate. This is observed development, with no new inference. Saved float16 maps are illustrative; original full-float metrics remain authoritative. Figures require visual QA before manuscript integration.</p>'
    for page in pages:
        folder = html.escape(page['folder'])
        report += f'<section><h2>{int(page["budget"]*100)}% training masks — {page["models_and_views"]} systems/views</h2><h3>Every original case, shared GT-defined ROIs</h3><img src="{folder}/all8_shared_rois.png" alt="All eight original cases and all model views"><h3>Every original full native field</h3><img src="{folder}/all8_full_fields.png" alt="All eight full fields and every model view"><p><a href="{folder}/plan.json">Immutable input/display policy</a> · <a href="{folder}/fixed_case_metrics.csv">Original full-float case summaries</a></p></section>'
    OUT.mkdir(exist_ok=True)
    (OUT / 'report.html').write_text(report)
    save_json(OUT / 'index.json', {'at': now(), 'pages': pages, 'unique_source_images_across_budgets': 8,
              'saved_case_model_maps': 144, 'new_model_inference_or_fit': False,
              'visual_QA_pending': True, 'manuscript_integrated': False})
    print('Rendered complete source saved-map galleries:2 budgets;8 original images;144 maps; no new inference.', flush=True)


def render(directory, budget, ids, native, models):
    cmap = plt.get_cmap('magma').copy(); cmap.set_bad('#ccd4d8')
    for zoom, name in [(True, 'all8_shared_rois'), (False, 'all8_full_fields')]:
        guard()
        with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
            plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                                 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
            width, height = 1.75 * (2 + len(models)), 19.1
            fig, axes = plt.subplots(8, 2 + len(models), figsize=(width, height), squeeze=False)
            for i, identifier in enumerate(ids):
                image, mask = native[identifier]; left, top, right, bottom = roi(mask)
                crop = (slice(top, bottom), slice(left, right))
                axes[i, 0].imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                axes[i, 0].set_ylabel(f'Case {i+1}\n' + ('crack' if by_mask(mask) else 'normal'), fontsize=8)
                if zoom:
                    axes[i, 0].add_patch(plt.Rectangle((left, top), right-left, bottom-top, fill=False, edgecolor='white', lw=.8))
                    axes[i, 1].imshow(image[crop], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    outline(axes[i, 1], mask[crop])
                else:
                    axes[i, 1].imshow(np.where(mask == IGNORE, 3, mask),
                        cmap=ListedColormap(['#20292f', '#36dbd0', '#e9993c', '#ccd4d8']),
                        norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), interpolation='nearest')
                for j, model in enumerate(models, 2):
                    score = model['maps'][identifier][crop] if zoom else model['maps'][identifier]
                    truth = mask[crop] if zoom else mask
                    axes[i, j].imshow(np.ma.masked_where(truth == IGNORE, score), cmap=cmap,
                                      vmin=0, vmax=1, interpolation='nearest')
                    outline(axes[i, j], truth, score)
                for axis in axes[i]:
                    axis.set_xticks([]); axis.set_yticks([])
                    for spine in axis.spines.values(): spine.set_visible(False)
            titles = ['Native EL', 'EL + crack truth\nshared ROI' if zoom else 'Canonical\nannotation']
            titles += [LABELS[model['system']] for model in models]
            for axis, label in zip(axes[0], titles): axis.set_title(label, fontsize=8.5, pad=10)
            fig.subplots_adjust(left=.055, right=.99, top=.925, bottom=.095, wspace=.09, hspace=.18)
            fig.suptitle(f'All eight original SCDD source cases | {int(budget*100)}% train masks | fixed seed11',
                         x=.055, y=.987, ha='left', fontsize=15, color='#193e48')
            count = models[0]['training_masks']
            fig.text(.055, .96, f'{count} training masks +120 validation masks; upstream UCF574 exposure additional. Every applicable system/view retained.', fontsize=10)
            bar = fig.add_axes([.59, .057, .36, .009])
            fig.colorbar(ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap), cax=bar,
                         orientation='horizontal', ticks=[0, .25, .5, .75, 1]).set_label('Crack softmax score | fixed [0,1]', fontsize=9)
            fig.text(.055, .055, 'Cyan: crack truth. White dashed: display score >0.5. Gray: ignored.\nSame full fields / same GT-defined ROI for every model.', fontsize=9, color='#506d77')
            fig.text(.055, .017, 'Observed source validation selected checkpoints; these are not independent test images. Teacher1/ensemble share one fit.\n'
                     'Original saved float16 maps are illustrative; all numerical masks/metrics remain the original full-float evaluations. No new inference or case selection.',
                     fontsize=8.4, color='#506d77')
            for extension in ['png', 'pdf', 'svg']:
                fig.savefig(directory / f'{name}.{extension}', dpi=180, facecolor='white')
            plt.close(fig)


def by_mask(mask):
    return bool((mask == 1).any())


if __name__ == '__main__':
    main()
