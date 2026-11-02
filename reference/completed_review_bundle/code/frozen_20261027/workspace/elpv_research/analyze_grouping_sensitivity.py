"""Retrospective dependency sensitivity; keep registered primary bootstrap intact."""
from pathlib import Path
import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,now,digest
from elresearch.publication_statistics import paired_seed_group_bootstrap

BASE=ROOT/'artifacts/publication_20261017'
FOLDER=BASE/'grouping_sensitivity_v1'


def main():
    if (BASE/'final_confirmation_v1/observation_opened.json').exists():
        raise RuntimeError('Register this source-development sensitivity before final model observation')
    FOLDER.mkdir(exist_ok=True)
    original=read_json(BASE/'held_source_paired_uncertainty.json');stats=read_json(BASE/'statistics_plan_v2.json')
    plan={'version':1,'kind':'Retrospective conditional acquisition-family sensitivity; not a replacement primary analysis',
        'original_statistics_plan_sha256':sha256(BASE/'statistics_plan_v2.json'),
        'origin_review_sha256':sha256(BASE/'whole_frame_similarity_audit_v1/origin_review.json'),
        'source_family_union':['ARTS_00020','ARTS_00081'],'assumption':'Treat both provided capture families as potentially related because one training-image pair is highly similar. This does not establish a common physical module.',
        'budgets':[.2,1.],'comparators':['dinov3_frozen','resnet18_full','resnet50_full','efficientnet_b0_full'],
        'repetitions':stats['repetitions'],'seed':stats['seed'],'analysis_code_sha256':sha256(__file__),
        'bootstrap_implementation_sha256':sha256(ROOT/'elresearch/publication_statistics.py'),
        'scope':'Same saved development predictions, fitted models and source thresholds; point estimates and source studies unchanged; original and merged intervals both retained.'}
    if (FOLDER/'plan.json').exists() and read_json(FOLDER/'plan.json')!=plan:raise RuntimeError('Preserve previous grouping-sensitivity specification')
    if not (FOLDER/'plan.json').exists():save_json(FOLDER/'plan.json',plan)
    outputs=[]
    for budget in plan['budgets']:
        def predictions(family):
            values=[]
            for seed in [11,22,33]:
                folder=BASE/'scdd_development_v1/runs/holdout_ARTS'/f'budget_{budget:g}'/f'{family}_seed{seed}'
                run=read_json(folder/'run.json');target=read_json(folder/'target_development.json')
                if sha256(folder/'target_development_predictions.npz')!=target['prediction_sha256']:
                    raise RuntimeError('Development source predictions changed')
                with np.load(folder/'target_development_predictions.npz',allow_pickle=False) as f:
                    row={key:f[key].copy() for key in ['ids','y','groups','logits']}
                for group in plan['source_family_union']:
                    if not (row['groups']==group).any():raise RuntimeError('Specified source family is absent')
                # Cast to object: assigning a longer sensitivity ID must not be
                # silently truncated by a fixed-width numpy string dtype.
                groups=row['groups'].astype(object);groups[np.isin(groups,plan['source_family_union'])]='possible_related_ARTS20_ARTS81'
                row.update(groups=np.asarray(groups,dtype=str),seed=seed,threshold=run['source_calibrated_logit_threshold'])
                values.append(row)
            return values
        lora=predictions('dinov3_lora')
        for family in plan['comparators']:
            baseline=predictions(family)
            result=paired_seed_group_bootstrap(baseline,lora,stats['repetitions'],stats['seed'])
            parent=next(c for c in original['comparisons'] if c['fold']=='holdout_ARTS' and c['training_fraction']==budget and c['first']==family)
            for key,value in result['metrics'].items():
                if value['mean_seed_difference']!=parent['bootstrap']['metrics'][key]['mean_seed_difference']:
                    raise RuntimeError('A grouping sensitivity must not change point predictions/metrics')
            outputs.append({'fold':'holdout_ARTS','training_fraction':budget,'first':family,'second':'dinov3_lora',
                'original':parent['bootstrap'],'merged_possible_related_families':result})
            save_json(FOLDER/'results.json',{'at':now(),'plan_sha256':sha256(FOLDER/'plan.json'),
                'original_paired_results_sha256':sha256(BASE/'held_source_paired_uncertainty.json'),'comparisons':outputs,
                'status':'complete' if len(outputs)==8 else 'partial','scope':plan['scope'],'physical_module_identity_verified':False})
            print('Grouping sensitivity',budget,family,'AUROC interval',result['metrics']['auroc']['acquisition_family_percentile_ci95'],flush=True)


if __name__=='__main__':main()
