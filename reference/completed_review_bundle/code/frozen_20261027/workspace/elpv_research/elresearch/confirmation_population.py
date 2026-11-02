"""Frozen data-quality population views over one unchanged scored cohort.

No model scores enter the policy. The entire authorized pool is retained and
scored once; primary and sensitivity summaries use the same saved predictions.
"""
from .common import read_json
from . import confirmation_registry as gate


def checked_policy(binding):
    policy=read_json(gate.check_file(binding))
    if policy['version']!=1 or policy['before_confirmation_model_inference'] is not True:
        raise ValueError('Analysis population policy must precede target model inference')
    for value in policy['evidence']:gate.check_file(value)
    screen_binding=next((value for value in policy['evidence'] if value['path'].endswith('whole_frame_similarity_audit_v1/result.json')),None)
    review_binding=next((value for value in policy['evidence'] if value['path'].endswith('whole_frame_similarity_audit_v1/origin_review.json')),None)
    if not screen_binding or not review_binding:raise ValueError('Bind the actual similarity screen and pre-score origin review')
    screen=read_json(gate.check_file(screen_binding));review=read_json(gate.check_file(review_binding))
    if not screen['search_complete'] or screen['confirmation_model_inference'] or review['model_scores_used']:
        raise ValueError('Data-quality decisions cannot use confirmation model scores')
    for name,cohort in policy['cohorts'].items():
        if name not in gate.ALLOWED_COHORTS:raise ValueError('Population policy refers to an unauthorized cohort')
        manifest=read_json(gate.check_file(cohort['manifest']));partition=gate.ALLOWED_COHORTS[name][0]
        rows=[r for r in manifest['records'] if r['partition']==partition];ids={r['id'] for r in rows}
        if set(cohort['subsets'])!={'primary','full_v2_pool','broader_similarity_screen'}:
            raise ValueError('Keep the fixed primary/full-pool/broader-screen views')
        if cohort['subsets']['full_v2_pool']['excluded_ids']:
            raise ValueError('The full-pool sensitivity cannot drop images')
        if name.startswith('pvelad'):
            strong={v[key] for v in screen['strong_similarity_pairs'] for key in ['first_id','second_id']
                if v['first_id'].startswith('pvelad/') and v['second_id'].startswith('pvelad/')}
            broad={v[key] for v in screen['screened_pairs'] for key in ['first_id','second_id']
                if v['first_id'].startswith('pvelad/') and v['second_id'].startswith('pvelad/')}
            reviewed={v[key] for v in review['pvel_decisions'] for key in ['first_id','second_id']}
            if reviewed!=strong or set(cohort['subsets']['primary']['excluded_ids'])!=strong:
                raise ValueError('Primary exclusions must equal every reviewed strong-pair member, without outcome-selected extras')
            if set(cohort['subsets']['broader_similarity_screen']['excluded_ids'])!=broad:
                raise ValueError('Broader sensitivity must retain the complete prespecified screen membership')
        for subset in cohort['subsets'].values():
            excluded=subset['excluded_ids']
            if len(excluded)!=len(set(excluded)) or not set(excluded)<=ids:
                raise ValueError('Population exclusions must be unique authorized identities')
            retained=[r for r in rows if r['id'] not in set(excluded)]
            actual={'images':len(retained),'positive':sum(r['target'] for r in retained),
                'negative':sum(1-r['target'] for r in retained)}
            if actual!={k:subset[k] for k in actual} or not actual['positive'] or not actual['negative']:
                raise ValueError('Population counts or class support differ from frozen metadata')
    return policy


def selected_rows(rows,cohort_name,policy=None,subset='primary'):
    if not policy or cohort_name not in policy['cohorts']:
        if subset!='primary':raise ValueError('No sensitivity population declared for this cohort')
        return rows,[]
    value=policy['cohorts'][cohort_name]['subsets'][subset];excluded=value['excluded_ids']
    retained=[r for r in rows if r['id'] not in set(excluded)]
    return retained,excluded
