"""Source-metadata-only candidate inventory; never reads confirmation pixels/scores.

All completed development variants remain visible. Hyperparameter-search runs
enter only through their existing source-only selection records. Full-budget
deterministic duplicates are represented once rather than as three replications.
The resulting specification stays UNREVIEWED until all scientific source work is
complete; its existence cannot authorize test access.
"""
import re
from pathlib import Path

from elresearch.common import ROOT,read_json,save_json,sha256,now,digest
from elresearch.confirmation_workflow import STUDIES,specification_template

BASE=ROOT/'artifacts/publication_20261017'


def candidate_family(domain,fold,budget,spec,study,override=None):
    # Dense reference/position/evidence controls retain the parent's HPO family.
    # The actual architecture name and full specification identify the fitted
    # system; study identity also separates independently trained controls.
    method=override or spec.get('name') or spec.get('family')
    if not method:raise RuntimeError('Model family identity missing')
    method=re.sub(r'[^A-Za-z0-9_-]','_',method)
    identity=digest({'study':study,'specification':spec,'semantic_name':method})[:12]
    base=f"{domain}__{fold}__b{budget:g}__{method}__{identity}".replace('.','p')
    return method,base


def candidate_kind(run,folder):
    if (Path(folder)/'model.joblib').exists():return 'classical'
    model=run.get('model',{})
    if 'cached_augmented_features' in run or ('encoder' in model and 'architecture' not in model):return 'image'
    if 'architecture' in model:return 'dense'
    raise ValueError('Unknown completed source-model output contract')


def main():
    source_index=read_json(BASE/'source_development_identity_v1.json')
    dataset={r['id']:r['dataset'] for r in source_index['records']}
    candidates=[];seen={};unresolved=[];bindings=[]
    def selected_file(path):
        if not path.exists():unresolved.append(str(path.relative_to(ROOT)));return None
        bindings.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path)});return read_json(path)
    def add(path,study,name=None):
        path=Path(path)
        if not path.exists():unresolved.append(str(path.relative_to(ROOT)));return
        run=read_json(path);spec=run.get('model',run.get('specification',{}));seed=run['seed'];budget=run['training_budget_fraction']
        kind=candidate_kind(run,path.parent)
        if (kind=='classical' or spec.get('architecture')=='dinov3_normal_distance') and budget==1. and seed!=11:return
        ids=run['training_ids']+run['validation_ids']+run.get('unlabelled_source_ids',[])
        if not set(ids)<=set(dataset):raise RuntimeError('Candidate uses unaudited source identities')
        domains={dataset[i] for i in ids}
        if len(domains)!=1:raise RuntimeError('Unexpected mixed-domain source fitting')
        domain=domains.pop();fold='source' if domain=='ucf' else 'internal'
        for part in path.parts:
            if part.startswith('holdout_'):fold=part
        name,base=candidate_family(domain,fold,budget,spec,study,name)
        views=list(run.get('inference_views',{'primary':{}}))
        for view in views:
            family=base+('__'+view if view!='primary' else '')
            identifier=f'{family}_seed{seed}'
            if identifier in seen:
                if seen[identifier]!=str(path):raise RuntimeError('Different source fits collide in candidate identity')
                continue
            seen[identifier]=str(path)
            cohort_names=['scdd_publication_v1']
            if domain=='ucf' or (fold=='internal' and budget in [.2,1.]):
                cohort_names += ['ucf_reported_real_confirmation_v1','pvelad_restricted_confirmation_v2']
            role=('UCF-only source-selected frozen-weight transfer; UCF507 is within-dataset group-disjoint confirmation' if domain=='ucf' else
                  'SCDD group-disjoint test; UCF transfer is development-dataset-exposed; PVEL is restricted crack/author-normal evaluation' if fold=='internal' else
                  'Acquisition-held-out weights; primary analysis is matching SCDD test stratum, with other strata secondary; no cross-model pooled AUROC')
            declaration={'kind':kind,'family':family,'seed':seed,'source_view':view,
                'run_folder':str(path.parent.relative_to(ROOT)),'source_protocol':str((BASE/study/'protocol.json').relative_to(ROOT)),
                'cohorts':cohort_names,'analysis_stratum':fold.removeprefix('holdout_') if fold.startswith('holdout_') else 'whole',
                'analysis_role':role}
            source=run.get('inference_views',{}).get(view,run)
            candidates.append({'id':identifier,'declaration':declaration,'source_dataset':domain,'fold':fold,
                'training_fraction':budget,'method':name,'seed':seed,'inference_view':view,'run_sha256':sha256(path),
                'checkpoint_declared_sha256':run['checkpoint_sha256'],
                'source_validation_metrics':source['validation_metrics'],
                'source_validation_localization':source.get('validation_localization'),
                'source_image_reload_report_present':(path.parent/'source_reload_verification.json').exists() if kind=='image' else None,
                'scope':'Candidate source metadata, not final eligibility or confirmation evidence'})
    # SCDD development families and UCF-only post-adaptation/source-reference fits.
    for _,study in STUDIES.items():
        for path in sorted((BASE/study/'runs').glob('**/run.json')):add(path,study)
    source=selected_file(BASE/'calibration_v1/selected_hyperparameters.json')
    if source:
        for family in source['selected']:
            for run in family['runs']:add(ROOT/run['path']/'run.json','calibration_v1',family['specification']['family'])
    source=selected_file(BASE/'classical_controls_v1/selected_hyperparameters.json')
    if source:
        for family in source['selected']:add(ROOT/family['source_model_path']/'run.json',STUDIES['classical'],family['specification']['family'])
    source=selected_file(BASE/'representation_scale_v1/selected_hyperparameters.json')
    if source:
        for family in source['selected']:add(ROOT/family['source_run_path']/'run.json',STUDIES['representation_scale'],family['specification']['family'])
    parent=selected_file(BASE/'dense_development_v1/selected_hyperparameters.json')
    if parent:
        protocol=read_json(BASE/'dense_development_v1/protocol.json')
        for spec in parent['selected']:
            candidate=next(c for c in protocol['calibration']['candidates'] if c['family']==spec['family'] and c['encoder_lr']==spec['encoder_lr'] and c['head_lr']==spec['head_lr'])
            for seed in [11,22,33]:add(BASE/'dense_development_v1/calibration'/f"{candidate['name']}_seed{seed}"/'run.json',STUDIES['dense'],spec['family'])
    source=selected_file(BASE/'mechanistic_controls_v1/selected_hyperparameters.json')
    if source:
        rate=source['selected_patch']['head_lr']
        for seed in [11,22,33]:add(BASE/'mechanistic_controls_v1/calibration'/f'dinov3_patch_mlp_lr{rate:g}_seed{seed}'/'run.json',STUDIES['mechanistic'],'dinov3_patch_mlp')
        add(BASE/'mechanistic_controls_v1/source_transfer/ucf_distance_seed11/run.json',STUDIES['mechanistic'],'dinov3_global_normal_distance')
    for path in sorted((BASE/'published_methods_v1/source_transfer').glob('*/run.json')):add(path,STUDIES['published_methods'])
    source=selected_file(BASE/'published_methods_v1/selected_hyperparameters.json')
    if source:
        protocol=read_json(BASE/'published_methods_v1/protocol.json')
        for selected in source['selected']:
            candidate=next(c for c in protocol['calibration']['candidates'] if c['semi']==selected['semi'] and c['learning_rate']==selected['learning_rate']
                and (not c['semi'] or (c['vat_xi']==selected['vat_xi'] and c['negative_threshold']==selected['negative_threshold'])))
            for seed in [11,22,33]:add(BASE/'published_methods_v1/calibration'/f"{candidate['name']}_seed{seed}"/'run.json',STUDIES['published_methods'],selected['name'])
    for path in sorted((BASE/'published_methods_v1/source_sensitivity').glob('*/run.json')):add(path,STUDIES['published_methods'])
    output=BASE/'confirmation_preparation_v1';output.mkdir(exist_ok=True)
    candidates.sort(key=lambda r:r['id'])
    catalog={'at':now(),'kind':'Unreviewed source-only candidate inventory; all complete variants visible, no test scoring',
        'source_selection_bindings':bindings,'source_identity_index_sha256':sha256(BASE/'source_development_identity_v1.json'),
        'candidates':candidates,'unresolved_source_selections_or_runs':sorted(set(unresolved)),
        'all_source_stages_complete':all(read_json(BASE/study/'status.json')['state']=='complete' for study in STUDIES.values()),
        'scope':'No confirmation pixels/scores read; source selection and every fitted checkpoint still require formal binding/review.'}
    save_json(output/'candidate_catalog.json',catalog)
    # A source reviewer may derive a NEW reviewed specification later; do not
    # silently overwrite an existing explicit scientific decision.
    template=output/'unreviewed_selection.json'
    if not template.exists():
        value=specification_template();value['models']=[c['declaration'] for c in candidates]
        save_json(template,value)
    print('Unreviewed source candidates',len(candidates),'unresolved selectors/runs',len(catalog['unresolved_source_selections_or_runs']),flush=True)
    return catalog

if __name__=='__main__':main()
