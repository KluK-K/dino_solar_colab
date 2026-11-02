"""One CPU-only pre-confirmation similarity screen; never changes source workers."""
import fcntl
from pathlib import Path
import os
import traceback
from importlib.metadata import version

from elresearch.common import ROOT,read_json,save_json,sha256,digest,now
from elresearch import near_duplicate_audit as audit


def main():
    folder=audit.FOLDER;folder.mkdir(exist_ok=True)
    with (folder/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (audit.BASE/'final_confirmation_v1/observation_opened.json').exists():
            raise RuntimeError('This is a pre-confirmation audit; no retroactive blind-audit claim')
        if read_json(audit.BASE/'inference_efficiency_v1/status.json')['state'] not in {'queued_after_all_development','complete'}:
            raise RuntimeError('Do not compete with formal inference timing')
        records,bindings,counts=audit.metadata()
        settings={'maximum_phash_distance':4,'maximum_relative_aspect_difference':.02,'minimum_gray_std':3.,
            'screen_correlation_64':.995,'strong_correlation_128':.9995,'strong_highpass_correlation':.995,'minimum_highpass_std':.25}
        plan={'version':1,'kind':'Nonlearned whole-frame near-copy screen before confirmation model scores',
            'counts':counts,'input_bindings':bindings,'settings':settings,
            'record_id_hash':digest([r['id'] for r in records]),
            'code_sha256':sha256(ROOT/'elresearch/near_duplicate_audit.py'),'runner_sha256':sha256(__file__),
            'versions':{name:version(name) for name in ['numpy','scipy','Pillow']},
            'scope':'Whole-frame resize/D4/positive-affine-photometric candidates; no crop search, learned model, automatic removals or physical-group invention.'}
        plan['audit_id']=digest(plan)[:20]
        path=folder/'plan.json'
        if path.exists() and read_json(path)!=plan:raise RuntimeError('Registered audit changed; preserve it and use an explicit versioned amendment')
        if not path.exists():
            save_json(path,plan);snap=folder/'source_snapshot';snap.mkdir()
            import shutil
            shutil.copy2(ROOT/'elresearch/near_duplicate_audit.py',snap/'near_duplicate_audit.py');shutil.copy2(__file__,snap/Path(__file__).name)
        save_json(folder/'input_records.json',{'records':records,'input_bindings':bindings,'role':'Data integrity only; not model evaluation inputs'})
        save_json(folder/'launch_state.json',{'at':now(),'pid':os.getpid(),'audit_id':plan['audit_id']})
        gray,hashes,shapes=audit.fingerprints(records,folder)
        result=audit.search(records,gray,hashes,shapes,settings,folder)
        print('CPU whole-frame screen complete',result['images'],'images;',len(result['strong_similarity_pairs']),'near-copy-compatible candidates requiring review',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        audit.FOLDER.mkdir(exist_ok=True)
        save_json(audit.FOLDER/'failure.json',{'at':now(),'error':str(error),'traceback':traceback.format_exc(),'confirmation_model_inference':False})
        raise
