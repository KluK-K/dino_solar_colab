import pytest

from elresearch.pvelad_data import read_annotation


def xml(label='crack',boxes=True):
    obj=f'<object><name>{label}</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>' if boxes else ''
    return '<annotation><filename>img000001.jpg</filename><size><width>10</width><height>10</height></size>'+obj+'</annotation>'


def test_voc_pixel_edges_and_crack_type_sensitivity_are_explicit(tmp_path):
    path=tmp_path/'annotation.xml';path.write_text(xml('star_crack'))
    result=read_annotation(path)
    assert result['boxes'][0]['normalized_xyxy']==[0.,0.,1.,1.]
    assert result['target']==1 and result['target_line_crack'] is None
    path.write_text(xml('crack'));assert read_annotation(path)['target_line_crack']==1


def test_missing_boxes_and_unknown_categories_are_not_implicitly_normal(tmp_path):
    path=tmp_path/'annotation.xml';path.write_text(xml(boxes=False))
    with pytest.raises(ValueError,match='normal-label'):read_annotation(path)
    path.write_text(xml('new_unverified_category'))
    with pytest.raises(ValueError,match='Unmapped'):read_annotation(path)
    path.write_text('<!DOCTYPE a [<!ENTITY x "y">]>'+xml())
    with pytest.raises(ValueError,match='DTD'):read_annotation(path)


def test_current_pvel_evaluation_cohort_does_not_turn_missing_crack_boxes_into_negatives():
    from elresearch.common import ROOT,read_json
    manifest=read_json(ROOT/'data/pvelad_restricted_confirmation_v2/manifest.json')
    for row in manifest['records']:
        assert '/test/' not in row['image_member']
        if row['target']==0:
            assert row['role']=='author_normal_negative' and '/othertypes/good/' in row['image_member']
        else:
            assert row['role']=='known_annotated_crack_positive'
            assert any(b['class'] in {'crack','star_crack'} for b in row['boxes'])
    assert all(r['target'] is None for r in manifest['unknown_records'])
    assert len(manifest['unknown_records'])==2075


def test_restricted_pvel_has_no_d4_repeated_identities_or_invented_physical_groups():
    from elresearch.common import ROOT,read_json
    rows=read_json(ROOT/'data/pvelad_restricted_confirmation_v2/manifest.json')['records']
    assert len({r['d4_pixel_sha256'] for r in rows})==len(rows)
    assert all(r['module_id'] is None and r['acquisition_id'] is None and r['split_group'] is None for r in rows)
