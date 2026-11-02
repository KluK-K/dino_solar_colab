"""Nonlearned whole-frame similarity audit; not a model evaluation.

Screens exact-D4/resize/positive-affine-photometric near copies. Similarity is
evidence for review, not proof of common physical origin. Cropped overlaps,
unseen acquisition repeats and pretraining membership are outside this audit.
"""
from pathlib import Path
from functools import lru_cache
from io import BytesIO
import json

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter

from .common import ROOT, read_json, save_json, sha256, digest, now

BASE=ROOT/'artifacts/publication_20261017'
FOLDER=BASE/'whole_frame_similarity_audit_v1'


def image_path(record):
    path=(ROOT/record['image_path']).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in [ROOT/'data',ROOT.parent/'elpv_midterm/data/source']):
        raise PermissionError('Similarity audit is restricted to authorized research/course image roots')
    return path


def d4(image):
    for k in range(4):
        value=np.rot90(image,k)
        yield value
        yield np.fliplr(value)


def phash(image):
    coefficients=dctn(np.asarray(image,dtype=float)-np.mean(image),norm='ortho')[:8,:8].ravel()
    flags=coefficients>np.median(coefficients[1:]);flags[0]=False
    return np.sum(flags.astype(np.uint64)<<np.arange(64,dtype=np.uint64),dtype=np.uint64)


def fingerprint(image):
    gray=np.asarray(image.convert('L').resize((64,64),Image.Resampling.LANCZOS),dtype=np.uint8)
    hashes=np.array([phash(v) for v in d4(gray)],dtype=np.uint64)
    return gray,hashes


def correlation(first,second,minimum_std=0.):
    a=np.asarray(first,dtype=np.float64).ravel();b=np.asarray(second,dtype=np.float64).ravel()
    a=a-a.mean();b=b-b.mean();sa=np.sqrt(np.mean(a*a));sb=np.sqrt(np.mean(b*b))
    if sa<=minimum_std or sb<=minimum_std:return None
    return float(np.clip(np.mean(a*b)/(sa*sb),-1.,1.))


def relevant_pair(a,b):
    # Same known acquisition family inside one analysis partition is already a
    # grouped unit. Still inspect any source/confirmation edge regardless.
    if a['audit_role']=='source' and b['audit_role']=='confirmation':return True
    if b['audit_role']=='source' and a['audit_role']=='confirmation':return True
    group_a=a.get('split_group');group_b=b.get('split_group')
    if (a['dataset']==b['dataset'] and a['audit_partition']==b['audit_partition']
        and group_a is not None and group_b is not None and group_a==group_b):return False
    return True


def metadata():
    index=read_json(BASE/'source_development_identity_v1.json');expected={r['id'] for r in index['records']}
    source={};bindings=[{'path':str((BASE/'source_development_identity_v1.json').relative_to(ROOT)),
        'sha256':sha256(BASE/'source_development_identity_v1.json')}]
    for binding in index['source_manifests']:
        path=ROOT/binding['path']
        if sha256(path)!=binding['sha256']:raise RuntimeError('Audited source manifest changed')
        bindings.append(binding)
        for row in read_json(path)['records']:
            if row['id'] in expected:source[row['id']]={**row,'audit_role':'source','audit_partition':row['dataset']+'/'+row['partition']}
    if set(source)!=expected or len(source)!=1045:raise RuntimeError('All1045 observed source identities required')
    records=list(source.values());counts={'source':1045}
    for name,partition,number in [('scdd_publication_v1','test',117),('ucf_reported_real_confirmation_v1','external',507),
                                  ('pvelad_restricted_confirmation_v2','external',12326),('elpv','development',1968)]:
        path=ROOT/'data'/name/'manifest.json';bindings.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
        rows=[r for r in read_json(path)['records'] if r['partition']==partition]
        if len(rows)!=number:raise RuntimeError('Declared audit cohort changed')
        role='auxiliary_development' if name=='elpv' else 'confirmation';counts[name]=len(rows)
        records.extend({**r,'audit_role':role,'audit_partition':name+'/'+partition} for r in rows)
    records.sort(key=lambda r:r['id'])
    if len(records)!=15963 or len({r['id'] for r in records})!=len(records):raise RuntimeError('Audit identities must be unique')
    return records,bindings,counts


def scan_phash_pairs(hashes,records,maximum_distance=4,block_size=64,start=0):
    """Unique i<j pairs; eight relative D4 alignments, no canonical-hash shortcut."""
    hashes=np.asarray(hashes,dtype=np.uint64)
    if hashes.shape!=(len(records),8):raise ValueError('Expected8 D4 hashes per image')
    for lo in range(start,len(records),block_size):
        hi=min(lo+block_size,len(records))
        distances=np.bitwise_count(hashes[lo:hi,0,None,None]^hashes[None,:,:]).min(axis=-1)
        pairs=[]
        for offset in range(hi-lo):
            i=lo+offset
            for j in np.flatnonzero(distances[offset]<=maximum_distance):
                if j>i and relevant_pair(records[i],records[j]):pairs.append((i,int(j)))
        yield lo,hi,pairs


def low_resolution_match(a,b,hash_a,hash_b,shape_a,shape_b,settings):
    distances=np.bitwise_count(np.uint64(hash_a)^hash_b)
    best=None
    for orientation,value in enumerate(d4(b)):
        if distances[orientation]>settings['maximum_phash_distance']:continue
        height,width=shape_b if (orientation//2)%2==0 else shape_b[::-1]
        ratio=(shape_a[1]/shape_a[0])/(width/height)
        if abs(np.log(ratio))>np.log(1+settings['maximum_relative_aspect_difference']):continue
        score=correlation(a,value,settings['minimum_gray_std'])
        if score is not None and score>=settings['screen_correlation_64'] and (best is None or score>best['correlation_64']):
            best={'orientation':orientation,'phash_distance':int(distances[orientation]),'correlation_64':score}
    return best


@lru_cache(maxsize=512)
def gray128(path,expected_sha256):
    path=Path(path)
    if sha256(path)!=expected_sha256:raise RuntimeError('Image changed during similarity audit')
    with Image.open(path) as image:
        return np.asarray(image.convert('L').resize((128,128),Image.Resampling.LANCZOS),dtype=np.float64)


def verify_pair(a,b,match,settings):
    first=gray128(str(image_path(a)),a['image_sha256']);second=list(d4(gray128(str(image_path(b)),b['image_sha256'])))[match['orientation']]
    ncc=correlation(first,second,settings['minimum_gray_std'])
    first_high=first-gaussian_filter(first,2.);second_high=second-gaussian_filter(second,2.)
    high=correlation(first_high,second_high,settings['minimum_highpass_std'])
    strong=(ncc is not None and high is not None and ncc>=settings['strong_correlation_128'] and high>=settings['strong_highpass_correlation'])
    return {'first_id':a['id'],'second_id':b['id'],'first_role':a['audit_role'],'second_role':b['audit_role'],
        'first_partition':a['audit_partition'],'second_partition':b['audit_partition'],
        'first_group':a.get('split_group'),'second_group':b.get('split_group'),**match,
        'correlation_128':ncc,'highpass_correlation_128':high,'near_copy_compatible':bool(strong),
        'same_exact_D4_pixels':bool(a.get('d4_pixel_sha256') is not None and a.get('d4_pixel_sha256')==b.get('d4_pixel_sha256')),
        'annotation_targets_differ':a['target']!=b['target'],
        'interpretation':'Similarity candidate only; no automatic origin proof, cohort removal or physical-group claim.'}


def fingerprints(records,folder):
    folder=Path(folder);parts=folder/'fingerprint_parts';parts.mkdir(exist_ok=True)
    gray=[];hashes=[];shapes=[]
    for start in range(0,len(records),256):
        selected=records[start:start+256];path=parts/f'{start:05d}.npz';proof=path.with_suffix('.json')
        signature=digest([{'id':r['id'],'image_sha256':r['image_sha256']} for r in selected])
        if path.exists() and proof.exists():
            old=read_json(proof)
            if old['input_signature']!=signature or sha256(path)!=old['payload_sha256']:raise RuntimeError('Partial image-hash cache changed')
            with np.load(path,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['ids'],[r['id'] for r in selected]);g=f['gray64'].copy();h=f['hashes'].copy();s=f['shapes'].copy()
        else:
            arrays=[];values=[];native=[]
            for row in selected:
                path_image=image_path(row);payload=path_image.read_bytes()
                import hashlib
                if hashlib.sha256(payload).hexdigest()!=row['image_sha256']:raise RuntimeError('Research image changed before audit')
                with Image.open(BytesIO(payload)) as image:
                    native.append([image.height,image.width]);small,codes=fingerprint(image);arrays.append(small);values.append(codes)
            g=np.array(arrays);h=np.array(values);s=np.array(native,dtype=np.int32)
            np.savez_compressed(path,ids=np.array([r['id'] for r in selected]),gray64=g,hashes=h,shapes=s)
            save_json(proof,{'input_signature':signature,'payload_sha256':sha256(path)})
        gray.append(g);hashes.append(h);shapes.append(s)
        save_json(folder/'status.json',{'at':now(),'state':'fingerprinting','completed_images':min(start+256,len(records)),'planned_images':len(records)})
    return np.concatenate(gray),np.concatenate(hashes),np.concatenate(shapes)


def search(records,gray,hashes,shapes,settings,folder):
    folder=Path(folder);progress=folder/'search_progress.json';journal=folder/'screened_pairs.jsonl'
    done=read_json(progress) if progress.exists() else {'next_row':0,'phash_candidate_pairs':0,'low_resolution_matches':0,'near_copy_compatible_pairs':0,'journal_bytes':0}
    if journal.exists():
        if journal.stat().st_size<done['journal_bytes']:raise RuntimeError('Similarity journal is shorter than committed progress')
        if journal.stat().st_size>done['journal_bytes']:
            # Only uncommitted tail of this own derived journal is replayed.
            with journal.open('r+b') as f:f.truncate(done['journal_bytes'])
    elif done['journal_bytes']:raise RuntimeError('Similarity journal is missing')
    for start,stop,pairs in scan_phash_pairs(hashes,records,settings['maximum_phash_distance'],64,done['next_row']):
        matches=[]
        for i,j in pairs:
            match=low_resolution_match(gray[i],gray[j],hashes[i,0],hashes[j],shapes[i],shapes[j],settings)
            if match is not None:matches.append(verify_pair(records[i],records[j],match,settings))
        with journal.open('a') as f:
            for value in matches:f.write(json.dumps(value,allow_nan=False)+'\n')
            f.flush()
        done.update(next_row=stop,phash_candidate_pairs=done['phash_candidate_pairs']+len(pairs),
            low_resolution_matches=done['low_resolution_matches']+len(matches),
            near_copy_compatible_pairs=done['near_copy_compatible_pairs']+sum(v['near_copy_compatible'] for v in matches),journal_bytes=journal.stat().st_size)
        save_json(progress,done);save_json(folder/'status.json',{'at':now(),'state':'similarity_search','planned_images':len(records),**done})
    pairs=[json.loads(line) for line in journal.read_text().splitlines()]
    save_json(folder/'result.json',{'at':now(),'plan_sha256':sha256(folder/'plan.json'),'images':len(records),**done,
        'screened_pairs':pairs,'strong_similarity_pairs':[v for v in pairs if v['near_copy_compatible']],
        'search_complete':True,'human_origin_review_complete':False,'confirmation_model_inference':False,
        'limits':['Not a proof of original-image or physical-component identity. No new cohorts, groups or labels assigned automatically.',
            'Whole-frame D4/resizing/positive-affine brightness sensitivity only; crop overlaps and upstream foundation pretraining membership untested.',
            'Low-information frames and below-threshold similarities can evade this screen. Absence of flagged pairs is not absence of all related images.',
            'ELPV auxiliary task overlaps do not by themselves imply a main crack model was fitted on ELPV labels. Report model-specific exposure.']})
    save_json(folder/'status.json',{'at':now(),'state':'complete_screen_origin_review_pending','images':len(records),**done})
    return read_json(folder/'result.json')
