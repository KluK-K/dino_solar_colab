import copy

import numpy as np
import pytest

from analyze_dense_development_uncertainty import saved_record
from elresearch.confirmation_analysis import endpoint_value, paired_endpoint, EndpointUnavailable


def inputs():
    ids=['negative_small','crack_a','negative_large','crack_b'];groups=['capture1','capture1','capture2','capture2']
    rows=[]
    for identifier,group,known,crack,pred,fp,structure,structure_fp in zip(ids,groups,
        [20,20,200,20],[0,4,0,4],[2,4,0,3],[2,1,0,0],[10,None,100,None],[2,None,0,None]):
        tp=pred-fp;dice=2*tp/(crack+pred) if crack+pred else 1.
        rows.append({'id':identifier,'acquisition_family':group,'known_pixels':known,'crack_pixels':crack,
            'predicted_crack_pixels':pred,'false_positive_pixels':fp,'crack_positive':bool(crack),
            'dice':dice,'iou':tp/(crack+pred-tp) if crack+pred-tp else 1.,'pixel_threshold':.5,
            'negative_image_normal_structure_pixels':structure,
            'negative_image_normal_structure_false_positive_pixels':structure_fp})
    run={'seed':11,'training_budget_fraction':.2,'source_logit_threshold':.1}
    evaluated={'validation_localization':{'per_image':rows,'pooled_pixel_average_precision':.23,
        'mean_positive_image_dice':(.75+6/7)/2,'pooled_negative_structure_false_positive_fraction':2/110}}
    arrays={'ids':np.array(ids),'groups':np.array(groups),'y':np.array([0,1,0,1]),'logits':np.array([-.5,.5,-1.,.1])}
    return run,evaluated,arrays,ids,groups


def adapt(values):
    run,evaluated,arrays,ids,groups=values
    return saved_record(run,evaluated,arrays,'fixture_dense',ids,groups)


def test_saved_native_adapter_preserves_image_weighting_strict_cutoff_and_pooled_ap():
    record=adapt(inputs())
    assert endpoint_value(record,'positive_image_Dice')==pytest.approx((.75+6/7)/2)
    # Structure area differs ten-fold; this is a ratio of pixel counts weighted
    # by resampled images/families, not a mean of image FPRs or pixel bootstrap.
    assert endpoint_value(record,'normal_structure_pixel_FPR')==pytest.approx(2/110)
    assert endpoint_value(record,'normal_structure_pixel_FPR',weight=np.array([2,2,1,1]))==pytest.approx(4/120)
    assert endpoint_value(record,'recall_at_source_threshold')==.5
    result=paired_endpoint([record],[record],'pooled_native_pixel_AP')
    assert result['first_mean']==.23 and result['mean_pair_difference']==0
    assert result['acquisition_family_percentile_ci95'] is None


def test_saved_native_adapter_rejects_identity_truth_group_and_count_drift():
    original=inputs()
    changes=[lambda x:x[2].update(ids=x[2]['ids'][::-1]),
        lambda x:x[2]['y'].__setitem__(0,1),
        lambda x:x[1]['validation_localization']['per_image'][0].update(acquisition_family='other'),
        lambda x:x[1]['validation_localization']['per_image'][1].update(dice=.1),
        lambda x:x[1]['validation_localization']['per_image'][0].update(negative_image_normal_structure_false_positive_pixels=3)]
    for change in changes:
        values=copy.deepcopy(original);change(values)
        with pytest.raises(ValueError):adapt(values)


def test_absent_structure_support_remains_not_estimable_without_substitution():
    values=inputs();local=values[1]['validation_localization']
    for row in local['per_image']:
        if not row['crack_positive']:
            row.update(negative_image_normal_structure_pixels=0,negative_image_normal_structure_false_positive_pixels=0)
    local['pooled_negative_structure_false_positive_fraction']=None
    record=adapt(values)
    assert endpoint_value(record,'normal_structure_pixel_FPR') is None
    with pytest.raises(EndpointUnavailable):paired_endpoint([record],[record],'normal_structure_pixel_FPR')
