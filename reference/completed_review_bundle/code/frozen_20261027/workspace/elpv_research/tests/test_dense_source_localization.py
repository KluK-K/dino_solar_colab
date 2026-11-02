import json
import numpy as np
import pytest
from analyze_dense_source_localization import seed_summary


def sample():
    rows=[]
    for identifier,known,cracks,dice,ap in [('p1',4000,20,0.,.01),('p2',1000,20,.5,.4),('p3',100,3,.5,.7),('n',100,0,1.,None)]:
        rows.append({'id':identifier,'known_pixels':known,'crack_pixels':cracks,'dice':dice,
            'annotated_crack_fraction':cracks/known,'pixel_average_precision':ap,
            'negative_image_false_positive_fraction':.01 if not cracks else None})
    return {'per_image':rows}


def test_native_overlap_is_not_replaced_by_image_detection_and_strict_ties_stay_negative():
    local=sample();values,rows=seed_summary(local,['p1','p2','p3','n'],[1,1,1,0],[.4,.3,.6,.8],.3)
    low,medium,high,whole,negative=values
    assert [r['images'] for r in [low,medium,high]]==[1,1,1]
    assert low['image_detected_with_zero_overlap_fraction']==1 and low['mean_positive_image_Dice']==0
    assert medium['source_image_recall']==0 and medium['zero_overlap_fraction']==0
    assert whole['mean_positive_image_AP']==pytest.approx((.01+.4+.7)/3)
    assert whole['image_detected_with_zero_overlap_images']==1 and negative['source_image_FPR']==1
    assert negative['mean_positive_image_AP'] is None and negative['mean_negative_image_pixel_FP_fraction']==.01


def test_empty_footprint_group_is_null_and_native_identity_or_label_errors_are_refused():
    local=sample();local['per_image']=local['per_image'][2:]
    values,_=seed_summary(local,['p3','n'],[1,0],[.6,.8],.3)
    restored=json.loads(json.dumps(values,allow_nan=False))
    assert restored[0]['images']==0 and restored[0]['zero_overlap_fraction'] is None
    assert restored[1]['mean_positive_image_Dice'] is None
    with pytest.raises(ValueError,match='label/native'):seed_summary(local,['p3','n'],[0,0],[.6,.8],.3)
    with pytest.raises(ValueError,match='aligned'):seed_summary(local,['p3','p3'],[1,0],[.6,.8],.3)
