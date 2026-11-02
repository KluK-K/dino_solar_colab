"""Retrospective held-source DEVELOPMENT errors: native annotations, not saliency.

All main predictions already exist. No new encoder call, target threshold fit,
mask-derived training feature, or sealed confirmation image is allowed here.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

from elresearch.common import ROOT, read_json, save_json, sha256, digest, now
from elresearch.data import canonical_mask, IGNORE
from elresearch.publication_train import probability_metrics

BASE = ROOT/'artifacts/publication_20261017'
OUTPUT = BASE/'image_failure_analysis_v1'
FAMILIES = ['dinov3_frozen', 'dinov3_lora', 'dinov3_full', 'resnet18_full', 'resnet50_full', 'efficientnet_b0_full']
SHORT = ['Frozen', 'LoRA', 'Full', 'R18', 'R50', 'Eff']


def json_seed_records(frame):
    """Absent subgroup support stays null, never NaN or a manufactured zero."""
    return frame.astype(object).where(pd.notnull(frame), None).to_dict('records')


def characteristics(row, permitted, root=ROOT):
    if row['id'] not in permitted or row['partition'] not in {'train', 'validation'} or row['dataset'] != 'scdd':
        raise PermissionError('Failure analysis permits only predeclared471 observed SCDD development identities')
    path = Path(root)/row['mask_path']
    if sha256(path) != row['mask_sha256']:
        raise RuntimeError('Development annotation changed')
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
    mask, ambiguous = canonical_mask(raw, 'scdd'); known = mask != IGNORE
    if (ambiguous and row['target']==0) or not known.any() or int((mask == 1).any()) != row['target']:
        raise ValueError('Development mask/target semantics differ')
    if raw.ndim==3: raw=raw[...,0]
    crack = int((mask == 1).sum()); denominator = int(known.sum())
    fraction = crack/denominator
    area_bin = ('negative' if crack == 0 else 'positive_le_0.5pct' if fraction <= .005 else
                'positive_0.5_to_2pct' if fraction <= .02 else 'positive_gt_2pct')
    structure = np.isin(raw, [4, 9]) & known
    return {'id': row['id'], 'target': row['target'], 'known_pixels': denominator, 'crack_pixels': crack,
        'annotated_crack_fraction_of_known_pixels': fraction, 'area_bin': area_bin,
        'normal_structure_pixels': int(structure.sum()), 'normal_structure_fraction_of_known_pixels': float(structure.sum()/denominator),
        'other_defect_pixels': int((mask == 2).sum()), 'negative_has_other_annotated_defect': bool(crack == 0 and (mask == 2).any()),
        'negative_has_structure': bool(crack == 0 and structure.any())}


def prediction_bundle(cell, rows):
    ids = None; y = None; outputs = {}; calibrations = []; bindings = []
    for family in FAMILIES:
        values = []
        model = next(v for v in cell['models'] if v['family'] == family)
        for item in model['seeds']:
            path = ROOT/item['run']; folder = path.parent
            if sha256(path) != item['run_sha256'] or sha256(folder/'target_development_predictions.npz') != item['prediction_sha256']:
                raise RuntimeError('Completed development inputs changed')
            run = read_json(path); target = read_json(folder/'target_development.json')
            if set(run['training_ids']+run['validation_ids']) & set(target['identity']['target_ids']):
                raise PermissionError('Source-target identity leakage')
            with np.load(folder/'target_development_predictions.npz', allow_pickle=False) as predictions:
                current_ids = predictions['ids'].copy(); current_y = predictions['y'].copy(); logits = predictions['logits'].copy()
            if ids is None:
                ids, y = current_ids, current_y
            np.testing.assert_array_equal(current_ids, ids); np.testing.assert_array_equal(current_y, y)
            np.testing.assert_array_equal(current_y, [rows[i]['target'] for i in current_ids])
            np.testing.assert_array_equal(current_ids, target['identity']['target_ids'])
            raw = probability_metrics(y, logits); calibrated = probability_metrics(y, logits, run['source_temperature'])
            if raw != target['calibration_raw'] or calibrated != target['calibration_source_temperature']:
                raise RuntimeError('Calibration metrics do not match saved predictions')
            values.append({'seed': run['seed'], 'logits': logits, 'prediction': logits > run['source_calibrated_logit_threshold'],
                'source_threshold': run['source_calibrated_logit_threshold'], 'source_temperature': run['source_temperature']})
            calibrations.append({'held_source': cell['target_acquisition_source'], 'training_fraction': cell['training_fraction'],
                'family': family, 'seed': run['seed'], 'raw': raw, 'source_temperature': calibrated,
                'temperature_fitted_on_target': False})
            bindings.append({'run': str(path.relative_to(ROOT)), 'run_sha256': sha256(path), 'predictions_sha256': item['prediction_sha256']})
        outputs[family] = values
    return {'ids': ids, 'y': y, 'outputs': outputs, 'calibrations': calibrations, 'bindings': bindings}


def subgroup_records(cell, bundle, metadata):
    labels = np.array([metadata[i]['area_bin'] for i in bundle['ids']])
    records = []
    subsets = {name: labels == name for name in ['positive_le_0.5pct', 'positive_0.5_to_2pct', 'positive_gt_2pct']}
    negative = bundle['y'] == 0
    subsets.update({'all_positives': bundle['y'] == 1, 'all_negatives': negative,
        'negative_other_defect': np.array([metadata[i]['negative_has_other_annotated_defect'] for i in bundle['ids']]),
        'negative_no_other_defect': negative & np.array([not metadata[i]['negative_has_other_annotated_defect'] for i in bundle['ids']]),
        'negative_with_structure': np.array([metadata[i]['negative_has_structure'] for i in bundle['ids']]),
        'negative_without_structure': negative & np.array([not metadata[i]['negative_has_structure'] for i in bundle['ids']])})
    for family, fits in bundle['outputs'].items():
        for fit in fits:
            for name, keep in subsets.items():
                n = int(keep.sum()); positive = int(bundle['y'][keep].sum())
                correct = int((fit['prediction'][keep] == bundle['y'][keep]).sum()) if n else 0
                records.append({'held_source': cell['target_acquisition_source'], 'training_fraction': cell['training_fraction'],
                    'family': family, 'seed': fit['seed'], 'subgroup': name, 'images': n, 'positive': positive,
                    'correct': correct, 'incorrect': n-correct, 'correct_fraction': correct/n if n else None,
                    'rate_meaning': 'crack recall' if name.startswith('positive') or name == 'all_positives' else 'noncrack specificity',
                    'source_threshold': fit['source_threshold'], 'threshold_fitted_on_target': False})
    return records


def choose_examples(cell, bundle, metadata):
    """One positive/negative per source, using fixed documented error/annotation ties."""
    wrong = np.sum(np.array([v['prediction'] != bundle['y'] for v in bundle['outputs']['dinov3_lora']]), axis=0)
    result = []
    for label in [1, 0]:
        candidates = [i for i, y in enumerate(bundle['y']) if y == label]
        def key(index):
            m = metadata[bundle['ids'][index]]
            annotation_tie = (m['annotated_crack_fraction_of_known_pixels'] if label else
                              -m['normal_structure_fraction_of_known_pixels'])
            return (-int(wrong[index]), annotation_tie, str(bundle['ids'][index]))
        index = min(candidates, key=key); identifier = str(bundle['ids'][index])
        errors = {family: int(sum(v['prediction'][index] != label for v in fits)) for family, fits in bundle['outputs'].items()}
        result.append({'id': identifier, 'held_source': cell['target_acquisition_source'], 'training_fraction': .2,
            'target': label, 'LoRA_incorrect_seeds': int(wrong[index]), 'all_model_incorrect_seeds': errors,
            'candidate_images': len(candidates), 'max_LoRA_incorrect_seeds_in_class': int(wrong[candidates].max()),
            'characteristics': metadata[identifier], 'scope': 'Selected development example, not random or representative.'})
    return result


def figure(examples, rows):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 10.6))
    colors = {'crack': '#c83e68', 'structure': '#e6a33e', 'other': '#37a6a0', 'excluded': '#8f929a'}
    def rgb(color): return np.array(matplotlib.colors.to_rgb(color))
    for row_number, source in enumerate(['ARTS', 'CFVS', 'CSIR']):
        for pair, target in enumerate([1, 0]):
            item = next(x for x in examples if x['held_source'] == source and x['target'] == target)
            row = rows[item['id']]
            if sha256(ROOT/row['image_path']) != row['image_sha256']: raise RuntimeError('Development image changed')
            with Image.open(ROOT/row['image_path']) as image: gray = np.asarray(image.convert('L')).copy()
            with Image.open(ROOT/row['mask_path']) as image: raw = np.asarray(image).copy()
            mask = canonical_mask(raw, 'scdd')[0]; overlay = np.zeros((*mask.shape, 4))
            if raw.ndim==3: raw=raw[...,0]
            for flag, color, alpha in [(mask == IGNORE, colors['excluded'], .25),
                (mask == 2, colors['other'], .5), (np.isin(raw, [4, 9]), colors['structure'], .5),
                (mask == 1, colors['crack'], .95)]:
                overlay[flag, :3] = rgb(color); overlay[flag, 3] = alpha
            for column in range(2):
                ax = axes[row_number, 2*pair+column]; ax.imshow(gray, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if column: ax.imshow(overlay, interpolation='nearest')
                ax.axis('off')
                ax.set_title(('EL image' if column == 0 else 'Native annotation'), fontsize=9, color='#264853', pad=6)
            left = axes[row_number, 2*pair]
            right = axes[row_number, 2*pair+1]
            left.text(0, 1.16, f"{source} · {'Annotated crack' if target else 'No annotated crack'}", transform=left.transAxes,
                fontsize=10, fontweight='bold', color='#163e4b')
            left.text(0, -.07, row['name'].removesuffix('.png'), transform=left.transAxes, fontsize=7.8, color='#415f6b')
            left.text(0, -.14, 'Incorrect / 3 seeds: '+ ' · '.join(f'{short} {item["all_model_incorrect_seeds"][family]}' for family, short in zip(FAMILIES, SHORT)),
                transform=left.transAxes, fontsize=6.9, color='#415f6b')
    fig.suptitle('Development failures shown against real annotations', x=.035, ha='left', y=.985,
        fontsize=16, color='#173f4b')
    handles = [Patch(facecolor=color, label=label) for color,label in [
        (colors['crack'], 'Crack'), (colors['structure'], 'Ribbons / busbars'),
        (colors['other'], 'Other intrinsic defect'), (colors['excluded'], 'Excluded task annotation')]]
    fig.legend(handles=handles, ncol=4, loc='upper left', bbox_to_anchor=(.025,.95), frameon=False, fontsize=9)
    fig.subplots_adjust(left=.035, right=.985, top=.86, bottom=.11, hspace=.48, wspace=.10)
    fig.text(.035,.037,'20% source-training labels. Selection: most LoRA seed errors, then smallest positive crack fraction / most negative structure, then ID.', fontsize=8, color='#56717c')
    fig.text(.035,.012,'Annotations are ground truth, not model attention or heatmaps. Examples are selected, not representative; structure co-occurrence does not establish the cause of an error.', fontsize=7.7, color='#56717c')
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUTPUT/f'development_failure_examples.{extension}', dpi=280, facecolor='white')
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True); held = read_json(BASE/'held_source_development_complete_cells.json')
    settings = read_json(BASE/'scdd_development_v1/protocol.json'); manifest = read_json(ROOT/'data/scdd_publication_v1/manifest.json')
    if len(held['complete_cells']) != 6: raise RuntimeError('All six predeclared cells must be complete')
    permitted = set(settings['permitted_target_development_ids'])
    rows = {r['id']: r for r in manifest['records'] if r['partition'] in {'train', 'validation'}}
    if len(rows) != 471 or permitted != set(rows): raise PermissionError('Observed development scope differs')
    plan = {'version': 2, 'scope': 'Retrospective exploratory development-error analysis after aggregate image results; not confirmatory or causal.',
        'export_amendment': read_json(OUTPUT/'amendment.json'),
        'complete_cells_digest': digest(held['complete_cells']), 'source_protocol_sha256': sha256(BASE/'scdd_development_v1/protocol.json'),
        'manifest_sha256': sha256(ROOT/'data/scdd_publication_v1/manifest.json'), 'code_sha256': sha256(__file__),
        'canonical_mask_code_sha256': sha256(ROOT/'elresearch/data.py'),
        'area_bins_fraction_of_known_pixels': [0., .005, .02, 1.],
        'example_selection': 'At20% for each source and class: highest LoRA incorrect-seed count; positive smallest annotated crack fraction, negative largest normal-structure fraction; then lexicographic ID. All6 systems reported.',
        'limits': ['Area is an annotation convention, not a physical crack width or electrical severity.',
            'All471 development masks characterize evaluation-only strata; masks do not become image-model inputs or budgeted training supervision.',
            'Crack-negative means no annotated crack, not physically healthy. No causal texture or busbar claim.',
            'Tiny subgroups and same capture families require transparent counts; no pixel pseudo-replication.',
            'No original117 test/UCF507/PVEL/ELPV test images or scores opened.']}
    if (OUTPUT/'plan.json').exists() and read_json(OUTPUT/'plan.json') != plan:
        raise RuntimeError('Preserve previous exploratory failure plan; use an explicitly versioned amendment')
    if not (OUTPUT/'plan.json').exists(): save_json(OUTPUT/'plan.json', plan)
    metadata = {identifier: characteristics(row, permitted) for identifier,row in rows.items()}
    subgroup = []; calibrations = []; examples = []; bindings = []
    for cell in held['complete_cells']:
        bundle = prediction_bundle(cell, rows); subgroup.extend(subgroup_records(cell, bundle, metadata))
        calibrations.extend(bundle['calibrations']); bindings.extend(bundle['bindings'])
        if cell['training_fraction'] == .2: examples.extend(choose_examples(cell, bundle, metadata))
    summaries = []
    for key, group in pd.DataFrame(subgroup).groupby(['held_source','training_fraction','family','subgroup'], sort=True):
        values = group['correct_fraction'].dropna().to_numpy()
        summaries.append(dict(zip(['held_source','training_fraction','family','subgroup'], key),
            images=int(group['images'].iloc[0]), positive=int(group['positive'].iloc[0]), rate_meaning=group['rate_meaning'].iloc[0],
            mean_correct_fraction=float(values.mean()) if len(values) else None,
            fitted_seed_sample_sd=float(values.std(ddof=1)) if len(values)>1 else None,
            all_three_seeds=json_seed_records(group)))
    result = {'at': now(), 'plan_sha256': sha256(OUTPUT/'plan.json'), 'subgroups': summaries,
        'calibration': calibrations, 'selected_examples': examples, 'source_bindings': bindings,
        'native_metadata': list(metadata.values()), 'limits': plan['limits'], 'confirmation_opened': False}
    save_json(OUTPUT/'results.json', result); pd.DataFrame(subgroup).to_csv(OUTPUT/'all_seed_subgroup_counts.csv', index=False)
    figure(examples, rows)
    print('Retrospective failure analysis:', len(subgroup), 'seed/subgroup records;', len(examples), 'selected annotated examples; all471 development images only')
    for source in ['ARTS','CFVS','CSIR']:
        for group in ['positive_le_0.5pct','positive_0.5_to_2pct','positive_gt_2pct','negative_other_defect','negative_no_other_defect']:
            row = next(x for x in summaries if x['held_source']==source and x['training_fraction']==.2 and x['family']=='dinov3_lora' and x['subgroup']==group)
            print(source, group, row['images'], row['rate_meaning'], row['mean_correct_fraction'])


if __name__ == '__main__': main()
