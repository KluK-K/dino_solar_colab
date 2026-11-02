"""Re-audit only previously observed source images; never open confirmation pixels."""
import numpy as np

from elresearch.common import ROOT, digest, now, read_json, save_json, sha256
from elresearch.confirmation_registry import bind_file
from elresearch.data import load_image
from elresearch.publication_data import orbit_hash


def main():
    folder=ROOT/'artifacts/publication_20261017'
    path=folder/'source_development_identity_v1.json'
    manifests=[ROOT/'data'/name/'manifest.json' for name in ['scdd_publication_v1','ucf']]
    bindings=[bind_file(p) for p in manifests]
    if path.exists():
        old=read_json(path)
        if old['source_manifests']!=bindings or digest({k:v for k,v in old.items() if k!='signature'})!=old['signature']:
            raise RuntimeError('Existing source identity index changed')
        print('Existing verified source identity index:',len(old['records']));return old
    rows=[]
    for manifest in manifests:
        for row in read_json(manifest)['records']:
            if row['partition'] not in {'train','validation'}: continue
            gray=np.array(load_image(row))
            orbit=orbit_hash(gray)
            if row.get('d4_pixel_sha256',orbit)!=orbit: raise RuntimeError('Previously audited source D4 identity changed')
            rows.append({k:row[k] for k in ['id','dataset','partition','split_group','image_sha256']})
            rows[-1]['d4_pixel_sha256']=orbit
    if len(rows)!=1045 or len({r['id'] for r in rows})!=1045:raise RuntimeError('Expected 471 SCDD and574 UCF observed source images')
    result={'created_at':now(),'kind':'Previously observed source identity index; no model scores or confirmation image reads',
            'source_manifests':bindings,'records':sorted(rows,key=lambda r:r['id']),
            'audit_code_sha256':sha256(__file__),
            'limits':['Acquisition families are conservative filenames, not verified physical modules.',
                      'The UCF source images have unverified individual real/simulated provenance.']}
    result['signature']=digest(result);save_json(path,result)
    print('Verified source identities:',len(rows));return result


if __name__=='__main__':main()
