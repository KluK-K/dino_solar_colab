"""Versioned publication cohorts. No image-model inference or test-set selection here."""
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .acquire import download_file
from .common import ROOT, digest, now, read_json, save_json, sha256
from .data import canonical_mask, IGNORE


def pixel_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def orbit_hash(array):
    """Exact invariance to the eight square rotations/reflections, without interpolation."""
    result=[]
    for flip in [False,True]:
        for k in range(4):
            x=np.rot90(array,k)
            if flip:x=np.fliplr(x)
            result.append(digest({'shape':list(x.shape),'pixels':pixel_hash(x)}))
    return min(result)


def summary(rows):
    result={}
    for source in sorted({r['acquisition_source'] for r in rows}):
        selected=[r for r in rows if r['acquisition_source']==source]
        result[source]={part:{'images':len(x:= [r for r in selected if r['partition']==part]),
                            'crack_positive':sum(r['target'] for r in x),
                            'capture_groups':len({r['split_group'] for r in x})}
                        for part in sorted({r['partition'] for r in selected})}
    return result


def audit_scdd():
    parent_path=ROOT/'data/scdd_confirmed_v1/manifest.json';parent=read_json(parent_path)
    rows=[];orbits=collections.defaultdict(list)
    for original in parent['records']:
        row=dict(original)
        for kind in ['image','mask']:
            if sha256(ROOT/row[kind+'_path'])!=row[kind+'_sha256']:
                raise ValueError(f'Changed {kind}: {row["id"]}')
        image=np.array(Image.open(ROOT/row['image_path']).convert('L'))
        mask,ambiguous=canonical_mask(np.array(Image.open(ROOT/row['mask_path'])),'scdd')
        if int((mask==1).any())!=row['target']:
            raise ValueError('Image-level crack label changed')
        if ambiguous and not row['target']:
            raise ValueError('Unresolved star-only annotation in the confirmation cohort')
        row['acquisition_source']=row['name'].split('_')[0]
        if row['acquisition_source'] not in {'ARTS','CFVS','CSIR'}:
            raise ValueError('Public-source overlap quarantine failed')
        row['d4_pixel_sha256']=orbit_hash(image)
        row['known_annotation_fraction']=float((mask!=IGNORE).mean())
        row['task']='crack_presence_in_annotated_image_field_of_view'
        row['provenance_level']='author_reported_real_private_acquisition_source'
        orbits[row['d4_pixel_sha256']].append(row['id']);rows.append(row)
    duplicates=[v for v in orbits.values() if len(v)>1]
    quarantine={i for cluster in duplicates for i in cluster}
    usable=[r for r in rows if r['id'] not in quarantine]
    signature=digest({'parent_manifest_sha256':sha256(parent_path),'records':usable,
                      'orbit_quarantine':duplicates,'audit_code_sha256':sha256(__file__)})
    result={'dataset':'scdd_publication_v1','signature':signature,'records':usable,
            'parent_dataset':'scdd_confirmed_v1','parent_signature':parent['signature'],
            'parent_manifest_sha256':sha256(parent_path),'created_at':now(),
            'source_commit':parent['source_commit'],'publication_eligible':True,
            'evaluation_state':'sealed','group_semantics':parent['group_semantics'],
            'partition_policy':'retain parent train/validation/test; quarantine all exact D4 duplicate identities',
            'task':'crack presence anywhere in annotated field of view, not central-cell electrical failure',
            'quarantine':duplicates,'source_counts':summary(usable),
            'limits':['Capture families are not confirmed physical module IDs.',
                      'Expert-curated convenience sample, not a prospective prevalence sample.',
                      'Images may include neighboring cells; labels describe the annotated field of view.',
                      'Data available from the author repository; raw images are not redistributed in our code release.'],
            'sources':['https://github.com/TheMakiran/BenchmarkELimages',
                       'https://doi.org/10.1016/j.sasc.2023.200048']}
    dest=ROOT/'data/scdd_publication_v1/manifest.json'
    if dest.exists():
        old=read_json(dest)
        if old['signature']!=signature:raise RuntimeError('Publication cohort is immutable')
        return old
    save_json(dest,result)
    save_json(ROOT/'artifacts/publication_20261017/scdd_data_audit.json',
              {k:v for k,v in result.items() if k!='records'})
    return result


def acquire_ucf_confirmation(workers=6):
    """Official test release, excluding all pilot-observed IDs and capture families.

    Real-image status is an inference from the author's test policy, not a sensor log.
    Unknown crack-negative annotations with ignored pixels remain excluded.
    """
    inventory=read_json(ROOT/'data/ucf/inventory.json')
    pilot=read_json(ROOT/'data/ucf/manifest.json')
    observed=[r for r in pilot['records'] if r['partition'] in {'train','validation'}]
    seen_ids={r['id'] for r in observed};seen_groups={r['group_id'] for r in observed}
    selected=[r for r in inventory['pairs'] if r['upstream_split']=='test'
              and r['id'] not in seen_ids and r['group_id'] not in seen_groups]
    folder=ROOT/'data/ucf_reported_real_confirmation_v1';folder.mkdir(parents=True,exist_ok=True)
    selection={'created_at':now(),'selection':'all official-release test IDs outside pilot-observed IDs/families',
               'source_commit':inventory['commit'],'observed_ids_sha256':digest(sorted(seen_ids)),
               'observed_groups_sha256':digest(sorted(seen_groups)),
               'selected_ids':[r['id'] for r in selected],'selected_groups':sorted({r['group_id'] for r in selected}),
               'selection_uses_model_scores':False}
    old_selection=folder/'selection.json'
    if old_selection.exists():
        old=read_json(old_selection)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in selection.items() if k!='created_at'}:
            raise RuntimeError('Confirmation selection changed')
    else:save_json(old_selection,selection)
    def download(pair):
        row=dict(pair)
        for kind in ['image','mask']:
            suffix=Path(row[kind+'_source']).suffix
            dest=ROOT/'data/ucf/raw'/kind/(hashlib.sha256(row['id'].encode()).hexdigest()+suffix)
            download_file(inventory['repo'],inventory['commit'],row[kind+'_source'],dest)
            row[kind+'_path']=str(dest.relative_to(ROOT))
        return row
    pairs=[];failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(download,r):r for r in selected}
        for future in concurrent.futures.as_completed(futures):
            try:pairs.append(future.result())
            except Exception as exc:failed.append({'id':futures[future]['id'],'reason':str(exc)})
            n=len(pairs)+len(failed)
            if n%50==0 or n==len(selected):
                save_json(folder/'download_status.json',{'complete':len(pairs),'requested':len(selected),
                          'failures':failed,'updated_at':now()})
                print('UCF confirmation data',n,'/',len(selected),'failures',len(failed),flush=True)
    if failed:raise RuntimeError(f'{len(failed)} downloads failed; selection retained for retry')
    upstream=ROOT/'references/publication_20261017/UCF_train.py'
    if not upstream.exists() or 'ignore_index=128' not in upstream.read_text():
        raise RuntimeError('Verify upstream ignore-index evidence before accepting index 128')
    observed_hashes={r['pixel_sha256'] for r in observed}
    scdd=read_json(ROOT/'data/scdd_publication_v1/manifest.json')
    source_orbits={r['d4_pixel_sha256'] for r in scdd['records']}
    rows=[];excluded=[];orbits=collections.defaultdict(list)
    for pair in sorted(pairs,key=lambda r:r['id']):
        row=dict(pair)
        image=np.array(Image.open(ROOT/row['image_path']).convert('L'))
        raw=np.array(Image.open(ROOT/row['mask_path']))
        if raw.ndim!=2 or raw.shape!=image.shape:
            excluded.append({'id':row['id'],'reason':'non-indexed or mismatched mask'});continue
        if not set(np.unique(raw))<={0,1,2,3,4,128}:
            excluded.append({'id':row['id'],'reason':'undocumented mask index'});continue
        positive=bool((raw==1).any());ignored=raw==128
        if ignored.any() and not positive:
            excluded.append({'id':row['id'],'reason':'crack-negative status unknown inside ignored annotation'});continue
        h=pixel_hash(image);orbit=orbit_hash(image)
        if h in observed_hashes or orbit in source_orbits:
            excluded.append({'id':row['id'],'reason':'exact observed-source or SCDD D4 image overlap'});continue
        row.update(dataset='ucf',cohort='ucf_reported_real_confirmation_v1',target=int(positive),
                   image_sha256=sha256(ROOT/row['image_path']),mask_sha256=sha256(ROOT/row['mask_path']),
                   pixel_sha256=h,d4_pixel_sha256=orbit,shape=list(image.shape),
                   crack_area_fraction=float((raw==1).mean()),known_annotation_fraction=float((~ignored).mean()),
                   split_group=row['group_id'],partition='external',acquisition_source='UCF',
                   real_or_synthetic='reported_real_from_release_test_policy',
                   provenance_level='inferred_from_author_manuscript_and_official_release_test_split',
                   annotation_type='indexed_pixel_mask_with_128_ignored',module_id=None)
        orbits[orbit].append(row['id']);rows.append(row)
    duplicated={i for cluster in orbits.values() if len(cluster)>1 for i in cluster[1:]}
    rows=[r for r in rows if r['id'] not in duplicated]
    excluded.extend({'id':i,'reason':'duplicate under exact D4 transforms'} for i in sorted(duplicated))
    signature=digest({'source_commit':inventory['commit'],'rows':rows,'excluded':excluded,
                      'selection':{k:v for k,v in selection.items() if k!='created_at'},'audit_code':sha256(__file__)})
    result={'dataset':'ucf_reported_real_confirmation_v1','signature':signature,'records':rows,'excluded':excluded,
            'created_at':now(),'source_commit':inventory['commit'],'evaluation_state':'sealed',
            'publication_eligible':False,
            'provenance_statement':'Authors report simulated images only in training; official README says the release train/test CSVs correspond to their evaluation. Test-image real status is therefore reported-policy inference, not independently verified file-level acquisition metadata.',
            'limits':['Release/publication total image counts differ.',
                      'Image identities and capture families seen in the old pilot are excluded, but physical module IDs remain unconfirmed.',
                      'Supplemental confirmation only until provenance is accepted; no pure-real performance claim depends on UCF.',
                      'Negative images may contain other defects and do not estimate fully healthy-cell false alarms.'],
            'source_counts':summary(rows),'upstream_ignore_code_sha256':sha256(upstream),
            'sources':['https://github.com/ucf-photovoltaics/UCF-EL-Defect',
                       'https://doi.org/10.1109/JPHOTOV.2021.3131059']}
    dest=folder/'manifest.json'
    if dest.exists():
        old=read_json(dest)
        if old['signature']!=signature:raise RuntimeError('Confirmation cohort changed')
        return old
    save_json(dest,result)
    save_json(ROOT/'artifacts/publication_20261017/ucf_confirmation_audit.json',
              {k:v for k,v in result.items() if k!='records'})
    return result


if __name__=='__main__':
    result=audit_scdd();print('SCDD audited',len(result['records']),result['source_counts'],flush=True)
    result=acquire_ucf_confirmation();print('UCF confirmation audited',len(result['records']),result['source_counts'],flush=True)
