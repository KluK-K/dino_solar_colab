"""Retrospective source diagnostic: image detection and native overlap can differ.

Uses complete saved selected source outputs only, with fixed registered pixel
threshold0.5 and previously used0.5%/2% annotation-footprint bins. No refit.
"""
import csv
import math

import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,now

BASE=ROOT/'artifacts/publication_20261017'
FOLDER=BASE/'dense_source_localization_diagnostic_v1'
SUBGROUPS=['positive_le_0.5pct','positive_0.5_to_2pct','positive_gt_2pct','all_positives','all_negatives']


def average(values):
    return float(np.mean(values)) if len(values) else None


def seed_summary(local,ids,y,logits,source_threshold):
    ids=np.asarray(ids);y=np.asarray(y);logits=np.asarray(logits)
    if len(set(ids.tolist()))!=len(ids) or y.shape!=logits.shape or y.shape!=ids.shape:
        raise ValueError('Source diagnostic requires unique aligned image outputs')
    if not math.isfinite(source_threshold) or not np.isfinite(logits).all():
        raise ValueError('Nonfinite source operating point')
    per_image=local['per_image'];by={r['id']:r for r in per_image}
    if len(by)!=len(per_image) or set(by)!=set(ids):raise ValueError('Native annotations/predictions are not aligned')
    images=[]
    for identifier,target,logit in zip(ids,y,logits):
        row=by[str(identifier)];count=row['known_pixels'];cracks=row['crack_pixels']
        if count<=0 or not 0<=cracks<=count or int(cracks>0)!=int(target):
            raise ValueError('Image label/native crack support differs')
        area=cracks/count
        if not np.isclose(area,row['annotated_crack_fraction'],rtol=0,atol=1e-15):
            raise ValueError('Use the canonical known-pixel denominator')
        if not 0<=row['dice']<=1:raise ValueError('Invalid native Dice')
        if target and row['pixel_average_precision'] is None:
            raise ValueError('Crack-positive native AP must be estimable')
        subgroup=('all_negatives' if not target else 'positive_le_0.5pct' if area<=.005 else
                  'positive_0.5_to_2pct' if area<=.02 else 'positive_gt_2pct')
        images.append({'id':str(identifier),'target':int(target),'area_fraction':area,'subgroup':subgroup,
            'image_detected':bool(logit>source_threshold),'zero_native_overlap':bool(target and row['dice']==0),
            'native_Dice':row['dice'],'positive_image_AP':row['pixel_average_precision'] if target else None,
            'negative_pixel_FP_fraction':row['negative_image_false_positive_fraction'] if not target else None})
    results=[]
    for subgroup in SUBGROUPS:
        selected=[r for r in images if (r['target']==1 if subgroup=='all_positives' else r['subgroup']==subgroup)]
        n=len(selected);positive=subgroup!='all_negatives'
        results.append({'subgroup':subgroup,'images':n,
            'source_image_recall':average([r['image_detected'] for r in selected]) if positive else None,
            'source_image_FPR':average([r['image_detected'] for r in selected]) if not positive else None,
            'mean_positive_image_Dice':average([r['native_Dice'] for r in selected]) if positive else None,
            'mean_positive_image_AP':average([r['positive_image_AP'] for r in selected]) if positive else None,
            'zero_overlap_images':sum(r['zero_native_overlap'] for r in selected) if positive else None,
            'zero_overlap_fraction':average([r['zero_native_overlap'] for r in selected]) if positive else None,
            'image_detected_with_zero_overlap_images':sum(r['image_detected'] and r['zero_native_overlap'] for r in selected) if positive else None,
            'image_detected_with_zero_overlap_fraction':average([r['image_detected'] and r['zero_native_overlap'] for r in selected]) if positive else None,
            'mean_negative_image_pixel_FP_fraction':average([r['negative_pixel_FP_fraction'] for r in selected]) if not positive else None})
    return results,images


def main():
    if (BASE/'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Retrospective source diagnostic must be fixed before final model observation')
    source=BASE/'dense_source_calibration_analysis_v1'
    source_plan=read_json(source/'plan.json');summary=read_json(source/'summary.json')
    if len(summary['models'])!=4:raise RuntimeError('Complete selected source systems required')
    bindings=[]
    for row in source_plan['source_bindings']:
        if 'run' not in row:continue
        path=ROOT/row['run']
        if sha256(path)!=row['sha256']:raise RuntimeError('Selected source run changed')
        bindings.append({'run':str(path.relative_to(ROOT)),'sha256':sha256(path),
                         'prediction_sha256':sha256(path.parent/'validation_predictions.npz')})
    if len(bindings)!=12:raise RuntimeError('Retain every selected architecture/seed pair')
    plan={'version':1,'kind':'Retrospective descriptive source-validation localization/detection diagnostic',
        'motivation':'Follow-up after viewing the complete selected source summaries and fixed native examples, including small-crack misses.',
        'all_selected_systems':[r['family'] for r in summary['models']],'seeds':[11,22,33],
        'positive_area_bins':[0,.005,.02,1],'area_definition':'Annotated crack pixels / known canonical native pixels; not physical severity.',
        'pixel_cutoff':.5,'image_cutoff':'Each original saved source-calibrated logit threshold, strict greater-than',
        'zero_overlap':'Crack-positive native Dice exactly0 at the fixed0.5 pixel threshold; does not necessarily mean no positive prediction anywhere.',
        'source_bindings':bindings,'source_selection_sha256':source_plan['source_selection_sha256'],
        'prior_footprint_analysis_sha256':sha256(BASE/'image_failure_analysis_v1/plan.json'),
        'analysis_code_sha256':sha256(__file__),
        'limits':['Reused source validation selects rates/checkpoints and image operating points; no independent or causal claim.',
            'Macro positive-image AP is a different endpoint from pooled AP, which also includes negative images; it is never substituted for pooled AP.',
            'Seed points are fitted-model records; repeated pixels and images across seeds are not independent experiments.',
            'No new model, raw image/mask read, source threshold fit, target calibration or confirmation model score.']}
    FOLDER.mkdir(exist_ok=True)
    if (FOLDER/'plan.json').exists() and read_json(FOLDER/'plan.json')!=plan:
        raise RuntimeError('Archive the previous source diagnostic before an amendment')
    if not (FOLDER/'plan.json').exists():save_json(FOLDER/'plan.json',plan)
    by_path={r['run']:r for r in bindings};groups=[];records=[]
    for family in plan['all_selected_systems']:
        fits=[]
        candidates=[(p,read_json(ROOT/p)) for p in by_path]
        for seed in plan['seeds']:
            path,run=next((p,r) for p,r in candidates if r['model']['family']==family and r['seed']==seed)
            folder=(ROOT/path).parent
            with np.load(folder/'validation_predictions.npz',allow_pickle=False) as saved:
                ids,y,logits=(saved[k].copy() for k in ['ids','y','logits'])
            local=run['validation_localization']
            if local['pixel_threshold']!=.5:raise RuntimeError('Do not replace fixed pixel threshold')
            values,images=seed_summary(local,ids,y,logits,run['source_logit_threshold'])
            whole=next(r for r in values if r['subgroup']=='all_positives')
            if not np.isclose(whole['mean_positive_image_Dice'],local['mean_positive_image_dice'],rtol=0,atol=1e-15):
                raise RuntimeError('Positive-image aggregation differs from original native evaluation')
            fits.append({'seed':seed,'pooled_native_pixel_AP':local['pooled_pixel_average_precision'],
                'subgroups':values,'per_image':images})
            records.extend({'family':family,'seed':seed,**value} for value in values)
        groups.append({'family':family,'fits':fits})
    result={'at':now(),'status':'complete','plan_sha256':sha256(FOLDER/'plan.json'),
        'families':groups,'seed_subgroup_records':len(records),'model_scores_from_confirmation':False,
        'scope':plan['limits']}
    save_json(FOLDER/'results.json',result)
    with (FOLDER/'all_seed_subgroups.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for family in groups:
        low=[next(r for r in fit['subgroups'] if r['subgroup']=='positive_le_0.5pct') for fit in family['fits']]
        whole=[next(r for r in fit['subgroups'] if r['subgroup']=='all_positives') for fit in family['fits']]
        print(family['family'],'low-footprint images',low[0]['images'],
            'mean zero-overlap fraction',average([r['zero_overlap_fraction'] for r in low]) if low[0]['images'] else None,
            'overall zero-overlap',average([r['zero_overlap_fraction'] for r in whole]),flush=True)


if __name__=='__main__':main()
