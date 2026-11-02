"""Native area and selected-example semantics; no real images or test cohorts."""
import numpy as np
import pytest
from PIL import Image
from elresearch.common import sha256
from analyze_image_failures import characteristics, choose_examples, FAMILIES, json_seed_records


def native_row(tmp_path, cracks=1):
    raw=np.zeros((20,20),np.uint8); raw[:,0]=5; raw[0,1:11]=4; raw[1,1:11]=9; raw[2,1:5]=11
    for i in range(cracks): raw[10+i//10,1+i%10]=10
    path=tmp_path/'fixture.png';Image.fromarray(raw).save(path)
    return {'id':'scdd/source-fixture','partition':'train','dataset':'scdd','target':int(cracks>0),
        'mask_path':str(path),'mask_sha256':sha256(path)},raw


def test_fraction_uses_known_annotation_denominator_and_positive_bins_are_disjoint(tmp_path):
    for count,expected in [(0,'negative'),(1,'positive_le_0.5pct'),(4,'positive_0.5_to_2pct'),(20,'positive_gt_2pct')]:
        row,raw=native_row(tmp_path,count);result=characteristics(row,{row['id']},tmp_path)
        assert result['known_pixels']==380 and result['crack_pixels']==count
        assert result['annotated_crack_fraction_of_known_pixels']==pytest.approx(count/380)
        assert result['area_bin']==expected and result['normal_structure_pixels']==20
        assert result['negative_has_other_annotated_defect']==(count==0)
    row,raw=native_row(tmp_path,1);raw[3,3]=28;Image.fromarray(raw).save(row['mask_path']);row['mask_sha256']=sha256(row['mask_path'])
    assert characteristics(row,{row['id']},tmp_path)['target']==1


def test_sealed_or_unauthorized_identity_is_refused_before_any_mask_access(monkeypatch):
    monkeypatch.setattr(Image,'open',lambda *a:pytest.fail('Must not open a sealed image/mask'))
    with pytest.raises(PermissionError): characteristics({'id':'test','partition':'test','dataset':'scdd'},{'test'})
    with pytest.raises(PermissionError): characteristics({'id':'unregistered','partition':'train','dataset':'scdd'},set())


def test_example_selection_does_not_invent_a_false_negative_when_no_seed_misses():
    ids=np.array(['p_small','p_large','n_plain','n_structured']);y=np.array([1,1,0,0])
    metadata={i:{'id':i,'annotated_crack_fraction_of_known_pixels':size,
                 'normal_structure_fraction_of_known_pixels':structure}
        for i,size,structure in zip(ids,[.001,.2,0.,0.],[0.,0.,0.,.3])}
    fits={family:[{'prediction':y.copy()} for _ in range(3)] for family in FAMILIES}
    bundle={'ids':ids,'y':y,'outputs':fits}
    values=choose_examples({'target_acquisition_source':'ARTS'},bundle,metadata)
    assert [r['id'] for r in values]==['p_small','n_structured']
    assert all(r['LoRA_incorrect_seeds']==0 for r in values)
    fits['dinov3_lora'][0]['prediction'][1]=0
    assert choose_examples({'target_acquisition_source':'ARTS'},bundle,metadata)[0]['id']=='p_large'


def test_absent_subgroup_support_serializes_as_null_without_inventing_a_zero_rate():
    import json
    import pandas as pd
    frame=pd.DataFrame([{'images':0,'correct_fraction':None},{'images':2,'correct_fraction':.5}])
    rows=json_seed_records(frame)
    restored=json.loads(json.dumps(rows,allow_nan=False))
    assert restored[0]['images']==0 and restored[0]['correct_fraction'] is None
    assert restored[1]['correct_fraction']==.5
