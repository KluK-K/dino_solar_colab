import hashlib

import numpy as np
from PIL import Image
import pytest
import torch

from elresearch.streaming_inference import stream_development_scores


def examples(tmp_path,count=8):
    rows=[]
    for i in range(count):
        path=tmp_path/f'{i}.png';Image.fromarray(np.full((18,37),i,dtype=np.uint8)).save(path)
        rows.append({'id':str(i),'image_path':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'partition':'validation','dataset':'fixture','target':i%2,'split_group':str(i)})
    return rows


def test_streaming_preserves_identity_native_geometry_and_bounds_prepared_batch(tmp_path):
    rows=examples(tmp_path);sizes=[];consumed=[]
    def scorer(batch):
        assert 'target' not in batch and 'mask' not in batch
        sizes.append(len(batch['images']))
        return {'image_logits':torch.tensor([float(x.mean()) for x in batch['gray_images']]),
                'crack_probability':torch.where(batch['valid'],.7,.95)}
    def consume(row,logit,native):
        assert native.shape==(18,37)
        np.testing.assert_allclose(native,.7,atol=1e-6)
        consumed.append(row['id'])
    result=stream_development_scores(rows,scorer,32,3,[r['id'] for r in rows],consume)
    np.testing.assert_array_equal(result['logits'],np.arange(8))
    assert sizes==[3,3,2] and result['maximum_prepared_images']==3
    assert result['native_maps_consumed']==8 and consumed==[r['id'] for r in rows]


def test_sealed_rows_are_rejected_before_any_image_or_model_is_read(tmp_path,monkeypatch):
    from elresearch import streaming_inference as module
    rows=examples(tmp_path,2);rows[-1]['partition']='external'
    monkeypatch.setattr(module,'load_image',lambda row:pytest.fail('A sealed cohort must be rejected before image access'))
    with pytest.raises(PermissionError,match='frozen confirmation'):
        stream_development_scores(rows,lambda batch:pytest.fail('No model call permitted'),32,1,[r['id'] for r in rows])


def test_streaming_rechecks_image_checksum(tmp_path):
    rows=examples(tmp_path,1);Image.fromarray(np.ones((18,37),dtype=np.uint8)).save(rows[0]['image_path'])
    with pytest.raises(ValueError,match='modified'):
        stream_development_scores(rows,lambda batch:pytest.fail('Changed image may not be inferred'),32,1,['0'])
