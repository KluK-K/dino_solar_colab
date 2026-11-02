"""Descriptive source-validation error subsets; held-out labels are never read."""
import numpy as np
from elresearch.common import ROOT, read_json, save_json, digest, now
from elresearch.data import records, load_mask


def main():
    train = records('ucf', 'train')
    fractions = []
    for row in train:
        mask = load_mask(row)
        if (mask == 1).any():
            fractions.append(float((mask == 1).sum() / (mask != 255).sum()))
    q1, q3 = np.quantile(fractions, [.25, .75])
    validation = records('ucf', 'validation')
    masks = {r['id']: load_mask(r) for r in validation}
    small, medium, large, no_crack, pure_normal = [], [], [], [], []
    for row in validation:
        mask = masks[row['id']]
        area = (mask == 1).sum() / (mask != 255).sum()
        if area > 0:
            (small if area <= q1 else medium if area <= q3 else large).append(row['id'])
        else:
            no_crack.append(row['id'])
            if not (mask == 2).any():
                pure_normal.append(row['id'])
    subsets = {'small_crack': small, 'medium_crack': medium, 'large_crack': large,
               'no_crack': no_crack, 'annotated_normal': pure_normal}
    runs = [read_json(p) for p in (ROOT / 'artifacts/dense_runs').glob('*/run.json')]
    runs = [r for r in runs if r['budget'] == 1. and r['epochs'] == 40]
    results = []
    for run in runs:
        folder = ROOT / 'artifacts/dense_runs' / run['run_id']
        with np.load(folder / 'validation_predictions.npz') as p:
            image_scores = dict(zip(p['ids'], p['scores']))
        summaries = {}
        normal_count = normal_false = other_count = other_false = 0
        for row in validation:
            mask = masks[row['id']]
            with np.load(folder / 'validation_maps' / (digest(row['id']) + '.npz')) as p:
                pred = p['score'].astype(np.float32) > .5
            normal_count += int((mask == 0).sum())
            normal_false += int((pred & (mask == 0)).sum())
            other_count += int((mask == 2).sum())
            other_false += int((pred & (mask == 2)).sum())
        for name, ids in subsets.items():
            dice, false_area = [], []
            for image_id in ids:
                with np.load(folder / 'validation_maps' / (digest(image_id) + '.npz')) as p:
                    score = p['score'].astype(np.float32)
                mask = masks[image_id]
                valid, gt = mask != 255, mask == 1
                pred = (score > .5) & valid
                denom = pred.sum() + gt.sum()
                dice.append(float(2 * (pred & gt).sum() / denom) if denom else 1.)
                false_area.append(float((pred & ~gt).sum() / valid.sum()))
            decisions = [image_scores[image_id] > run['threshold'] for image_id in ids]
            summaries[name] = {'n': len(ids), 'image_positive_rate': float(np.mean(decisions)) if ids else None,
                               'mean_crack_Dice': float(np.mean(dice)) if ids else None,
                               'mean_false_positive_pixel_fraction': float(np.mean(false_area)) if ids else None}
        results.append({'run_id': run['run_id'], 'architecture': run['architecture'],
                        'adaptation': run['adaptation'], 'reference': run['reference'], 'subsets': summaries,
                        'pooled_normal_pixel_false_positive_rate': normal_false / normal_count if normal_count else None,
                        'pooled_other_defect_pixel_false_positive_rate': other_false / other_count if other_count else None})
    output = {'created_at': now(),
              'scope': 'Post-hoc descriptive source-development analysis after viewing overall metrics; not a pre-registered confirmatory finding.',
              'area_definition': 'Crack-mask pixels / valid annotation pixels; not physical crack severity.',
              'training_positive_quartiles': [float(q1), float(q3)],
              'threshold_rule': 'Training positive quartiles define sizes; saved source validation image threshold; fixed pixel threshold 0.5.',
              'validation_subset_ids': subsets, 'runs': results}
    save_json(ROOT / 'artifacts/dense_failure_subsets.json', output)
    print({'training_quartiles': output['training_positive_quartiles'],
           'subset_counts': {k: len(v) for k, v in subsets.items()},
           'runs': [{'architecture': r['architecture'], 'adaptation': r['adaptation'], 'reference': r['reference'],
                     'small_recall': r['subsets']['small_crack']['image_positive_rate'],
                     'small_Dice': r['subsets']['small_crack']['mean_crack_Dice'],
                     'normal_false_pixels': r['subsets']['annotated_normal']['mean_false_positive_pixel_fraction']}
                    for r in results]})


if __name__ == '__main__':
    main()
