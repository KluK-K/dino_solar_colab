"""Known transformed copies and structured distractors; no research test images."""
import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

from elresearch import near_duplicate_audit as audit
from elresearch.common import sha256

SETTINGS={'maximum_phash_distance':4,'maximum_relative_aspect_difference':.02,'minimum_gray_std':3.,
    'screen_correlation_64':.995,'strong_correlation_128':.9995,'strong_highpass_correlation':.995,'minimum_highpass_std':.25}


def texture(seed=1):
    rng=np.random.default_rng(seed); yy,xx=np.indices((192,256))
    value=120+28*np.sin(xx/17)+17*np.cos(yy/13)+gaussian_filter(rng.normal(size=(192,256)),2)*80
    value[:,50:54]-=45;value[90:94,:]-=35
    return np.clip(value,0,255).astype(np.uint8)


def test_all_d4_resized_copies_retain_a_relative_orientation_match():
    source=texture();a,hash_a=audit.fingerprint(Image.fromarray(source))
    for value in audit.d4(source):
        image=Image.fromarray(value).resize((value.shape[1]*2,value.shape[0]*2),Image.Resampling.LANCZOS)
        b,hash_b=audit.fingerprint(image)
        result=audit.low_resolution_match(a,b,hash_a[0],hash_b,source.shape,(image.height,image.width),SETTINGS)
        assert result is not None and result['correlation_64']>.999


def test_unrelated_texture_and_low_information_are_not_origin_proofs():
    a,ha=audit.fingerprint(Image.fromarray(texture(1)));b,hb=audit.fingerprint(Image.fromarray(texture(24)))
    result=audit.low_resolution_match(a,b,ha[0],hb,(192,256),(192,256),SETTINGS)
    assert result is None
    assert audit.correlation(np.ones((4,4)),np.ones((4,4)),3.) is None
    assert audit.correlation(np.ones((4,4)),np.arange(16).reshape(4,4),3.) is None


def test_candidate_search_uses_relative_d4_hashes_and_never_invents_pvel_groups():
    full=np.uint64(2**64-1)
    hashes=np.array([[full]*8,[full^np.uint64(3)]*8],np.uint64)
    hashes[0,1]=np.uint64(0x0f0f0f0f0f0f0f0f);hashes[1,1]=np.uint64(0x3333333333333333)
    assert np.bitwise_count(hashes[0].min()^hashes[1].min())>4
    records=[{'dataset':'pvelad','audit_role':'confirmation','audit_partition':'pvel/external','split_group':None} for _ in range(2)]
    values=list(audit.scan_phash_pairs(hashes,records))
    assert values[0][2]==[(0,1)]
    same={'dataset':'scdd','audit_role':'source','audit_partition':'scdd/train','split_group':'provided1'}
    assert not audit.relevant_pair(same,same)
    assert audit.relevant_pair(same,{**same,'audit_role':'confirmation','audit_partition':'scdd/test'})


def test_high_resolution_verification_is_similarity_only_and_rejects_outside_roots(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path);data=tmp_path/'data';data.mkdir()
    original=Image.fromarray(texture());rotated=Image.fromarray(np.rot90(texture()))
    first=data/'first.png';second=data/'second.png';original.save(first);rotated.save(second)
    records=[{'id':str(i),'image_path':str(p),'image_sha256':sha256(p),'target':i,'dataset':'fixture',
        'audit_role':'source' if i==0 else 'confirmation','audit_partition':'fixture/'+str(i),'split_group':None}
        for i,p in enumerate([first,second])]
    a,ha=audit.fingerprint(original);b,hb=audit.fingerprint(rotated)
    match=audit.low_resolution_match(a,b,ha[0],hb,(original.height,original.width),(rotated.height,rotated.width),SETTINGS)
    verified=audit.verify_pair(*records,match,SETTINGS)
    assert verified['near_copy_compatible'] and verified['annotation_targets_differ']
    assert 'no automatic origin proof' in verified['interpretation']
    with pytest.raises(PermissionError):audit.image_path({'image_path':'/private/unrelated.png'})
