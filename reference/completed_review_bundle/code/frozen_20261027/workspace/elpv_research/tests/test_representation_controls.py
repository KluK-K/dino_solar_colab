import numpy as np
import pytest
import torch

from elresearch.classical_features import FeatureBank
from elresearch.classical_training import save_and_verify_probe,evaluate_target_development
from elresearch.representation_features import geometric_patch_weights,pool_dino_tokens,normalized_representation,cached_banks
from elresearch.representation_training import fit_representation_probe
from elresearch.publication_models import d4_view
from run_representation_scale_controls import feature_specifications,specification_grid


def test_patch_pooling_excludes_registers_padding_and_follows_d4_geometry():
    # Four patches, one valid. Huge CLS/register values must not enter the patch mean.
    hidden=torch.tensor([[[900.],[800.],[700.],[1.],[2.],[3.],[4.]]])
    weights=torch.tensor([[[0.,0.],[.5,0.]]])
    cls,mean=pool_dino_tokens(hidden,2,2,weights)
    assert cls.item()==900 and mean.item()==3
    patch=hidden[:,3:].reshape(1,2,2,1)
    for code in range(8):
        moved=d4_view(patch.permute(0,3,1,2),code).permute(0,2,3,1).reshape(1,4,1)
        transformed=torch.cat([hidden[:,:3],moved],dim=1)
        _,same=pool_dino_tokens(transformed,2,2,d4_view(weights[:,None],code)[:,0])
        assert same.item()==3
    with pytest.raises(ValueError,match='tokens'):pool_dino_tokens(hidden[:,:-1],2,2,weights)
    with pytest.raises(ValueError,match='no valid'):pool_dino_tokens(hidden,2,2,torch.zeros_like(weights))


def test_fractional_image_area_and_per_image_normalization_do_not_use_other_images():
    weights=geometric_patch_weights({'size':32,'top':8,'left':0,'h':16,'w':32},16)
    torch.testing.assert_close(weights,torch.full((2,2),.5))
    cls=np.array([[[1.,2.,4.]],[[100.,3.,9.]]],np.float32);mean=cls+1
    alone=normalized_representation(cls[:1],mean[:1],'concat')
    batch=normalized_representation(cls,mean,'concat')
    np.testing.assert_array_equal(alone,batch[:1])
    assert batch.shape==(2,1,6)
    np.testing.assert_allclose(batch.mean(-1),0,atol=1e-6)


def source_rows():
    return [{'id':str(i),'image_sha256':str(i),'pixel_sha256':str(i),'split_group':str(i),
        'partition':'train' if i<16 else 'validation','target':i%2} for i in range(24)]


@pytest.mark.parametrize('classifier',['logistic','linear_svm','rbf_svm'])
def test_frozen_probe_source_only_predictions_reload_and_target_guard(tmp_path,classifier):
    rows=source_rows();rng=np.random.default_rng(9)
    values=rng.normal(size=(24,8,6)).astype(np.float32)
    values[:,:,0]+=np.array([r['target'] for r in rows])[:,None]*3
    bank=FeatureBank(rows,values,{'kind':'synthetic source-only fixture'})
    spec={'classifier':classifier,'C':.1,'gamma_factor':1.,'family':'fixture'}
    model,result,scores=fit_representation_probe(spec,bank,rows[:16],rows[16:])
    assert list(model.named_steps)==['classifier']  # No secretly target-fitted scaler.
    saved=save_and_verify_probe(model,result,scores,bank,rows[16:],tmp_path)
    assert saved['source_reload_verified']
    assert saved['training_ids']==[r['id'] for r in sorted(rows[:16],key=lambda r:r['id'])]
    with pytest.raises(PermissionError):
        evaluate_target_development(model,saved,bank,[{**rows[-1],'partition':'test'}],{rows[-1]['id']},tmp_path)


def test_representation_grid_matches_control_axes_and_no_pseudo_seeds():
    features=feature_specifications();grid=specification_grid()
    assert len(features)==13 and len(grid)==49 and len({s['family'] for s in grid})==15
    for feature in features:
        candidates=[s for s in grid if s['feature_key']==feature['feature_key'] and s['classifier']=='logistic']
        assert [s['C'] for s in candidates]==[.01,.1,1.]
    assert {f['input_size'] for f in features if f['encoder']=='dinov3_small'}=={224,336,448}
    assert {f['pooling'] for f in features if f['encoder']=='dinov3_base'}=={'cls','patch_mean','concat'}


def test_feature_extractor_rejects_sealed_rows_before_model_access():
    row={'id':'sealed','partition':'test','image_sha256':'irrelevant'}
    with pytest.raises(PermissionError,match='confirmation'):
        cached_banks([row],{'name':'not loaded'},336,['cls'])


def test_cached_payload_tampering_is_rejected_before_probe_fitting(tmp_path,monkeypatch):
    from PIL import Image
    from elresearch import representation_features as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'feature_identity',lambda identity,size:{'encoder':identity,'size':size})
    monkeypatch.setattr(module,'load_image',lambda row:Image.new('L',(10,20)))
    class Encoder:
        def __init__(self,*args):pass
        def encode(self,images):
            values=np.tile(np.array([1.,2.,3.],np.float32),(len(images),8,1))
            return values,values+1
    monkeypatch.setattr(module,'FrozenRepresentationEncoder',Encoder)
    rows=source_rows()[:2]
    first=cached_banks(rows,{'name':'fixture','dimensions':3},32,['cls'],device='cpu')
    assert first['cls'].dimensions==3
    path=next((tmp_path/'cache/publication_representation_d4').glob('*.npz'))
    np.savez_compressed(path,cls=np.ones((8,3)),mean=np.ones((8,3)))
    with pytest.raises(RuntimeError,match='payload'):
        cached_banks(rows,{'name':'fixture','dimensions':3},32,['cls'],device='cpu')
