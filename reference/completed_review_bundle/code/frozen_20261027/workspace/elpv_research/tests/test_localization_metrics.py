import numpy as np
import pytest

from elresearch.localization_metrics import native_crack_metrics,pooled_native_metrics,box_point_localization


def test_unknown_pixels_do_not_count_as_false_positives_or_true_negatives():
    mask=np.zeros((4,4),np.uint8);mask[0,:]=255;mask[2,2]=1
    score=np.zeros((4,4));score[0,:]=1.;score[2,2]=.9
    result=native_crack_metrics(mask,score)
    assert result['dice']==1. and result['pixel_average_precision']==1.
    assert result['known_pixels']==12 and result['false_positive_pixels']==0


def test_many_empty_negatives_cannot_hide_a_missed_positive_dice():
    positive=np.zeros((4,4),np.uint8);positive[1,1]=1
    items=[{'mask':positive,'score':np.zeros((4,4))}]
    items.extend({'mask':np.zeros((4,4),np.uint8),'score':np.zeros((4,4))} for _ in range(9))
    result=pooled_native_metrics(items)
    assert result['mean_dice_all_images']==.9
    assert result['mean_positive_image_dice']==0.
    assert result['mean_negative_image_false_positive_fraction']==0.


def test_structure_false_positives_are_restricted_to_negative_images():
    mask=np.zeros((4,4),np.uint8);structure=np.zeros_like(mask,dtype=bool);structure[:,1]=True
    score=np.zeros((4,4));score[2,1]=.9
    result=native_crack_metrics(mask,score,normal_structure=structure)
    assert result['negative_image_normal_structure_false_positive_fraction']==.25
    mask[1,1]=1
    assert native_crack_metrics(mask,score,normal_structure=structure)['negative_image_normal_structure_pixels'] is None


def test_box_localization_uses_native_pixel_centers_and_reports_chance_coverage():
    score=np.zeros((4,4));score[1,2]=1
    result=box_point_localization(score,[[.5,.25,.75,.5]])
    assert result['maximum_point_in_target_box']
    assert result['box_union_pixel_center_fraction']==1/16
    assert not box_point_localization(score,[[0,0,.25,.25]])['maximum_point_in_target_box']


def test_metrics_reject_per_image_rescaled_or_mismatched_maps():
    with pytest.raises(ValueError,match='scale'):native_crack_metrics(np.zeros((3,3)),np.full((3,3),2.))
    with pytest.raises(ValueError,match='shapes'):native_crack_metrics(np.zeros((3,3)),np.zeros((4,4)))
    with pytest.raises(ValueError,match='target-class'):box_point_localization(np.zeros((3,3)),[])
