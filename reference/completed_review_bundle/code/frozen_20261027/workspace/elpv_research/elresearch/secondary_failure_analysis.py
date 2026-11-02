"""Frozen descriptive native-pixel failure strata; never change primary endpoints."""
import numpy as np

from .common import read_json
from . import confirmation_registry as gate
from analyze_dense_source_localization import seed_summary


def checked_failure_policy(binding):
    policy=read_json(gate.check_file(binding))
    if (policy['version']!=1 or policy['before_confirmation_model_inference'] is not True
        or policy['positive_area_bin_edges']!=[0,.005,.02,1]
        or policy['pixel_score_threshold']!=.5
        or policy['allowed_native_cohorts']!=['scdd_publication_v1','ucf_reported_real_confirmation_v1']):
        raise ValueError('Preserve source-fixed native failure bins, score cutoff and mask cohorts')
    evidence=policy['evidence']
    for value in evidence.values():gate.check_file(value)
    if gate.check_file(evidence['helper']).resolve()!=(gate.ROOT/'analyze_dense_source_localization.py').resolve():
        raise ValueError('Bind the executed native-failure helper, not an unrelated source file')
    plan=read_json(gate.check_file(evidence['source_plan']))
    source=read_json(gate.check_file(evidence['source_results']))
    if (source['plan_sha256']!=evidence['source_plan']['sha256'] or source['status']!='complete'
        or plan['analysis_code_sha256']!=evidence['helper']['sha256']
        or plan['positive_area_bins']!=policy['positive_area_bin_edges'] or plan['pixel_cutoff']!=.5
        or len(plan['source_bindings'])!=12
        or source['seed_subgroup_records']!=60 or len(source['families'])!=4
        or any(sorted(f['seed'] for f in model['fits'])!=[11,22,33] for model in source['families'])):
        raise ValueError('The completed source failure analysis differs from the declared evidence')
    return policy


def native_failure_summary(record,cohort_name,policy):
    if not policy or cohort_name not in policy['allowed_native_cohorts']:
        return None
    local=record.get('localization') or {}
    native=local.get('native_pixel_per_image',[])
    if not native:return None
    if local.get('pixel_threshold')!=policy['pixel_score_threshold']:
        raise ValueError('Frozen secondary failure analysis cannot replace the pixel operating point')
    permitted=set(str(x) for x in record['ids'])
    selected=[value for value in native if value['id'] in permitted]
    groups,images=seed_summary({'per_image':selected},record['ids'],record['y'],record['logits'],record['source_threshold'])
    architecture=record['model'].get('architecture')
    response=('Source-calibrated cosine-distance response;0.5 corresponds to the saved source noncrack-quantile cutoff, not a calibrated crack posterior.'
        if architecture=='dinov3_normal_distance' else 'Learned three-class crack softmax score at fixed0.5.')
    return {'cohort':cohort_name,'analysis_stratum':record['analysis_stratum'],
        'model_seed':record['model']['seed'],'analyzed_images':len(record['ids']),
        'positive_area_bin_edges':policy['positive_area_bin_edges'],'subgroups':groups,
        'image_detected_with_zero_overlap_ids':[r['id'] for r in images if r['image_detected'] and r['zero_native_overlap']],
        'pixel_response_semantics':response,'confidence_interval':None,
        'scope':'Frozen secondary descriptive subgroup counts/means. Primary comparisons unchanged; no target model, threshold, area-bin or case selection. Empty strata remain null. Pixels and repeated seeds are not independent samples.'}
