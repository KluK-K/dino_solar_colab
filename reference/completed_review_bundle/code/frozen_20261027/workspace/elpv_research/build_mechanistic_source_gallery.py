"""Retrospective full-budget source diagnostic from existing fixed native maps.

No model inference or new example search. Original eight cases and every
declared family remain visible. Ground truth appears only in its own column.
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

from build_dense_calibration_report import roi
from build_mechanistic_progress_readout import NAMES
from elresearch.common import ROOT, read_json, save_json, sha256, digest, now
from elresearch.data import records, load_image, load_mask, IGNORE

BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'mechanistic_source_native_gallery_v1'


def main():
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise RuntimeError('Source diagnostic must precede independent confirmation')
    state = read_json(BASE / 'inference_efficiency_v1/status.json')['state']
    if not (state.startswith('queued') or state in {'complete', 'failed', 'bounded_stop'}):
        raise RuntimeError('No source plotting during formal timings')
    index = read_json(BASE / 'mechanistic_complete_cells_v1/index.json')
    entry = next(e for e in index['complete_cells'] if e['fold'] == 'internal' and e['budget'] == 1.)
    cell_path = ROOT / entry['path']; cell = read_json(cell_path)
    if sha256(cell_path) != entry['sha256']:
        raise RuntimeError('Complete full-budget cell changed')
    by = {r['id']: r for r in records('scdd_publication_v1', 'validation')}
    families = [s['family'] for s in cell['all_model_summaries']]
    bindings = [{'path': entry['path'], 'sha256': entry['sha256']}]
    models = {}; original_ids = None; metrics = []
    for family in families:
        row = next(r for r in cell['all_original_fit_records'] if r['family'] == family and r['seed'] == 11)
        path = ROOT / row['source_run']; run = read_json(path)
        bindings.append({'path': row['source_run'], 'sha256': sha256(path)})
        ids = run['validation_localization']['fixed_example_ids']
        if original_ids is None: original_ids = ids
        if ids != original_ids or len(ids) != 8 or not set(ids) <= set(by):
            raise RuntimeError('Every family must retain the same original eight source cases')
        stats = {p['id']: p for p in run['validation_localization']['per_image']}
        prediction_path = path.parent / 'validation_predictions.npz'
        bindings.append({'path': str(prediction_path.relative_to(ROOT)), 'sha256': sha256(prediction_path)})
        with np.load(prediction_path, allow_pickle=False) as z:
            logits = dict(zip(z['ids'].tolist(), z['logits'].tolist()))
        maps = {}
        for identifier in ids:
            map_path = path.parent / 'fixed_validation_examples' / (digest(identifier) + '.npz')
            bindings.append({'path': str(map_path.relative_to(ROOT)), 'sha256': sha256(map_path)})
            with np.load(map_path, allow_pickle=False) as z:
                if str(z['id']) != identifier or z['score'].dtype != np.float16:
                    raise RuntimeError('Saved map identity/precision differs')
                value = z['score'].astype(np.float32)
            if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise RuntimeError('Display original finite scores on a fixed0..1 scale')
            maps[identifier] = value
            metrics.append({'id': identifier, 'family': family, 'seed': 11,
                'image_logit': logits[identifier], 'source_threshold': run['source_logit_threshold'],
                'image_flagged': logits[identifier] > run['source_logit_threshold'],
                'positive_image_Dice': stats[identifier]['positive_image_dice'],
                'original_predicted_crack_pixels': stats[identifier]['predicted_crack_pixels'],
                'original_crack_pixels': stats[identifier]['crack_pixels']})
        models[family] = {'maps': maps, 'stats': stats}
    native = {}
    for identifier in original_ids:
        row = by[identifier]
        if row['partition'] != 'validation': raise RuntimeError('Only observed source validation')
        for name in ['image', 'mask']:
            if sha256(ROOT / row[name + '_path']) != row[name + '_sha256']:
                raise RuntimeError('Original native image/mask bytes differ')
            bindings.append({'path': row[name + '_path'], 'sha256': row[name + '_sha256']})
        image, mask = np.asarray(load_image(row).convert('L')), load_mask(row)
        for family in families:
            s = models[family]['stats'][identifier]
            if (int((mask != IGNORE).sum()), int((mask == 1).sum())) != (s['known_pixels'], s['crack_pixels']):
                raise RuntimeError('Native truth support differs from original evaluation')
            if models[family]['maps'][identifier].shape != mask.shape or image.shape != mask.shape:
                raise RuntimeError('Native fields/maps/GT do not align')
        native[identifier] = (image, mask)
    negatives = sorted(i for i in original_ids if not by[i]['target'])
    positives = sorted((i for i in original_ids if by[i]['target']),
                       key=lambda i: (models[families[0]]['stats'][i]['annotated_crack_fraction'], i))
    if (len(negatives), len(positives)) != (3, 5): raise RuntimeError('Keep three original negatives and five positives')
    all_ids = negatives + positives; main_ids = [negatives[0], positives[0], positives[-1]]
    identity = {'builder_sha256': sha256(__file__), 'complete_cell': entry, 'bindings': bindings,
        'roi_helper_sha256': sha256(ROOT / 'build_dense_calibration_report.py'),
        'label_loader_sha256': sha256(ROOT / 'elresearch/data.py'), 'families': families,
        'all_original_ids': all_ids, 'main_ids': main_ids, 'seed': 11,
        'scope': 'Retrospective full-budget source follow-up; original seed11 convention and8fixedcases. No newly selected outcome-favorable cases. GT only in GT column; fixed0..1linear scores; no new inference.'}
    folder = OUT / digest(identity)[:20]; folder.mkdir(parents=True, exist_ok=True); result_path = folder / 'results.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['identity'] != identity: raise RuntimeError('Preserve native diagnostic identity')
        for a in result['figures']:
            if sha256(ROOT / a['path']) != a['sha256']: raise RuntimeError('Preserve prior native figures')
    else:
        assets = []; cmap = plt.get_cmap('magma').copy(); cmap.set_bad('#ccd4d8')
        labelmap = ListedColormap(['#252c32', '#36dbd0', '#e9993c', '#ccd4d8'])
        style = {'font.family': 'DejaVu Sans', 'font.size': 7., 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
        primary = ['deeplabv3plus_full', 'dinov3_dense_lora', 'dinov3_postlora_no_reference', 'dinov3_postlora_reference']
        specifications = [('shared_native_ROIs', main_ids, primary, True)]
        specifications += [(f'all8_raw_group{k//4+1}', all_ids, families[k:k+4], False) for k in range(0, len(families), 4)]
        for stem, identifiers, group, zoom in specifications:
            height = 5.6 if zoom else 11.1
            with plt.rc_context(matplotlib.rcParamsDefault), plt.rc_context(style):
                fig, axes = plt.subplots(len(identifiers), len(group)+2, figsize=(8.2, height), squeeze=False)
                for y, identifier in enumerate(identifiers):
                    image, mask = native[identifier]; left, top, right, bottom = roi(mask)
                    crop = (slice(top, bottom), slice(left, right)) if zoom else (slice(None), slice(None))
                    axes[y, 0].imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    if zoom:
                        axes[y, 0].add_patch(plt.Rectangle((left, top), right-left, bottom-top, fill=False, color='white', lw=.65))
                        axes[y, 1].imshow(image[crop], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                        truth = (mask[crop] == 1)
                        if truth.any() and not truth.all(): axes[y, 1].contour(truth, levels=[.5], colors=['#36dbd0'], linewidths=.7)
                    else:
                        axes[y, 1].imshow(np.where(mask == IGNORE, 3, mask), cmap=labelmap,
                            norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5], 4), interpolation='nearest')
                    axes[y, 0].set_ylabel(identifier.rsplit('/',1)[-1].removesuffix('.png').replace('_','\n',1), fontsize=6)
                    for col, family in enumerate(group, 2):
                        score = models[family]['maps'][identifier][crop]; known = mask[crop] != IGNORE
                        if zoom: axes[y, col].imshow(image[crop], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                        axes[y, col].imshow(np.ma.masked_where(~known, score), cmap=cmap, vmin=0, vmax=1,
                            interpolation='nearest', alpha=.65 if zoom else 1.)
                        pred = (score > .5) & known
                        if pred.any() and not pred.all(): axes[y, col].contour(pred, levels=[.5], colors=['white'], linewidths=.6, linestyles='--')
                    for ax in axes[y]:
                        ax.set_xticks([]); ax.set_yticks([])
                        for spine in ax.spines.values(): spine.set_visible(False)
                titles = ['Native EL', 'EL + GT\n(shared ROI)' if zoom else 'Canonical GT'] + [NAMES[f].replace(', ', ',\n').replace(' + ', '\n+ ') for f in group]
                for ax, title in zip(axes[0], titles): ax.set_title(title, fontsize=6.9, pad=7)
                fig.subplots_adjust(left=.08, right=.985, top=1-1.0/height, bottom=1.40/height, wspace=.12, hspace=.24)
                fig.suptitle('Full-budget source diagnostic | fixed seed11 cases', x=.08, ha='left', fontsize=9, y=.98, color='#234958')
                fig.text(.08, 1-.43/height, '351 training masks + 120 reused validation masks; upstream UCF 574 exposure is additional.', fontsize=7, color='#536f79')
                cax = fig.add_axes([.66, 1.02/height, .28, .07/height]);fig.colorbar(ScalarMappable(norm=plt.Normalize(0,1), cmap=cmap), cax=cax, orientation='horizontal', ticks=[0,.5,1]);cax.tick_params(labelsize=6)
                fig.text(.66, 1.18/height, 'Displayed crack score [0,1]', fontsize=6.4, color='#536f79')
                notes = ['GT cyan appears ONLY in the GT column; white model contours show saved score >0.5. Gray: ignored.',
                    'Same GT-defined ROI; fixed 0..1 score scale, overlay alpha 0.65. All 8 cases / all 11 families retained in raw groups.' if zoom else 'Raw full-field maps; same 0..1 linear scale. Orange GT: other defect. Normal-distance* is a calibrated response, not a crack posterior.',
                    'Saved float16 maps illustrate original seed11; quantitative endpoints use original full-float evaluations/all fitted seeds.',
                    'Outcome-motivated source diagnostic with unchanged original case/seed conventions; not independent confirmation.']
                for k, note in enumerate(notes): fig.text(.08, (.73-k*.20)/height, note, fontsize=6.4, color='#536f79')
                for ext in ['png','pdf','svg']:
                    p=folder/(stem+'.'+ext);fig.savefig(p,dpi=250,facecolor='white');assets.append({'path':str(p.relative_to(ROOT)),'sha256':sha256(p)})
                plt.close(fig)
        csv_path=folder/'all88_original_case_metrics.csv'
        with csv_path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
        result={'at':now(),'identity':identity,'figures':assets,'original_full_float_case_metrics':metrics,
                'csv':{'path':str(csv_path.relative_to(ROOT)),'sha256':sha256(csv_path)},
                'new_inference':False,'final_confirmation_access':False,
                'scope':'Source-only retrospective display of original fixed maps. GT not repeated on model panels. No quantitative metric recomputation from float16 display maps.'}
        save_json(result_path,result)
    save_json(OUT/'latest.json',{'path':str(result_path.relative_to(ROOT)),'sha256':sha256(result_path)})
    images=[a for a in result['figures'] if a['path'].endswith('.png')]
    (OUT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Full-budget source native diagnostic</title><style>body{max-width:1250px;margin:40px auto;font:16px/1.6 system-ui;color:#234958}img{width:100%}</style><h1>Full-budget source native diagnostic</h1><p>Original8fixedcases, all11families, originalseed11. Full-float quantitative results retain every fitted seed. Source validation selected checkpoints. GT is drawn only in its own column; white contours in model panels represent predictions. All outcomes concern provided annotation agreement.</p>'+''.join('<img src="'+folder.name+'/'+Path(a['path']).name+'">' for a in images)+'<p><a href="'+folder.name+'/all88_original_case_metrics.csv">Every original full-float case metric</a></p></html>')
    print('Rendered original88source maps:',result_path,'; no new inference')


if __name__=='__main__':main()
