"""Source-only UCF reference ablations for a clean frozen-weight transfer control."""
import argparse
import fcntl
import os
import shutil
import time
import traceback

from elresearch.common import ROOT,digest,now,read_json,save_json,sha256
from elresearch.publication_dense_train import FILES,scientific_hashes,train


BASE=ROOT/'artifacts/publication_20261017'
PARENT=BASE/'dense_development_v1'


def register(folder):
    parent=read_json(PARENT/'protocol.json');source=parent['calibration']
    proposal={'version':1,'kind':'UCF-only frozen-encoder normal-reference transfer controls',
        'registered_after':'SCDD image-only source-development results; before any new SCDD dense or independent confirmation scores',
        'rationale':'UCF labels calibrate SCDD hyperparameters, so SCDD-to-UCF is not a completely unseen selection domain. Preserve UCF-only fitted systems as a separately declared transfer direction.',
        'dataset':'ucf','source_manifest_sha256':source['manifest_sha256'],
        'source_training_ids':source['training_ids'],'source_validation_ids':source['validation_ids'],
        'upstream_dense_protocol':{'path':str((PARENT/'protocol.json').relative_to(ROOT)),'sha256':sha256(PARENT/'protocol.json')},
        'source_hyperparameter_selection':str((PARENT/'selected_hyperparameters.json').relative_to(ROOT)),
        'hyperparameter_rule':'Use the parent UCF-only frozen-DINO learning-rate winner; no SCDD metric chooses a learning rate, reference variant or checkpoint',
        'seeds':[11,22,33],'planned_runs':9,
        'systems':[{'name':'ucf_reference_position','geometry_weight':.1,'top_fraction':.05},
                   {'name':'ucf_reference_no_position','geometry_weight':0.,'top_fraction':.05},
                   {'name':'ucf_reference_mean_evidence','geometry_weight':.1,'top_fraction':1.}],
        'input_size':parent['input_size'],'batch_size':parent['batch_size'],
        'updates_per_round':parent['updates_per_round'],'maximum_rounds':parent['maximum_rounds'],
        'patience':parent['patience'],'weight_decay':parent['weight_decay'],
        'target_source_recall':parent['target_source_recall'],'device':parent['device'],
        'selection':parent['selection'],'scientific_hashes':scientific_hashes(),'runner_sha256':sha256(__file__),
        'primary_future_transfer_comparison':'All three reference variants versus the already declared UCF-only frozen DINO/LoRA/full/CNN source-calibration models; freeze comparisons before independent SCDD117 or PVEL scoring.',
        'annotation_accounting':'430 UCF training masks plus144 UCF validation masks; source provenance not individually verified. No SCDD/PVEL examples used for these fits.',
        'limits':['Source UCF real/simulated provenance is conditional, not audited per acquisition log.',
                  'These are source-development fits, not original-paper replications.',
                  'No automatic final confirmation; DINOv2 remains disabled.']}
    proposal['study_id']=digest(proposal)[:20];path=folder/'protocol.json'
    if path.exists():
        old=read_json(path)
        if digest({k:v for k,v in old.items() if k!='registered_at'})!=digest(proposal):raise RuntimeError('Registered UCF source-transfer control changed')
        return old
    snapshot=folder/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for name in FILES:shutil.copy2(ROOT/'elresearch'/name,snapshot/name)
    shutil.copy2(__file__,snapshot/'run_ucf_reference_transfer_controls.py')
    proposal['registered_at']=now();save_json(path,proposal);return proposal


def main():
    p=argparse.ArgumentParser();p.add_argument('--register-only',action='store_true')
    p.add_argument('--wait-for-dense-study',action='store_true');p.add_argument('--maximum-new-runs',type=int)
    args=p.parse_args();folder=BASE/'ucf_reference_transfer_v1';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=register(folder);print('UCF TRANSFER REGISTERED',settings['study_id'],'9 source-only fits',flush=True)
    if args.register_only:return
    completed,new=0,0
    try:
        while True:
            status=read_json(PARENT/'status.json')
            if status['state']=='complete':
                with (PARENT/'runner.lock').open('a') as prior:fcntl.flock(prior,fcntl.LOCK_EX)
                break
            if not args.wait_for_dense_study or status['state'] in {'failed','bounded_stop'}:
                raise RuntimeError('Parent dense study requires completion/resume: '+status['state'])
            save_json(folder/'status.json',{'state':'queued_after_dense_study','updated_at':now(),'pid':os.getpid(),
                'parent_state':status['state'],'planned_runs':9,'completed_runs':0})
            time.sleep(30)
        gpu_lock=(BASE/'gpu_work_queue.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
        if sha256(ROOT/'data/ucf/manifest.json')!=settings['source_manifest_sha256']:
            raise RuntimeError('Source UCF cohort changed')
        selected=read_json(ROOT/settings['source_hyperparameter_selection'])
        if selected['protocol_sha256']!=settings['upstream_dense_protocol']['sha256']:
            raise RuntimeError('Source-only hyperparameter lineage changed')
        source_spec=next(s for s in selected['selected'] if s['adaptation']=='frozen')
        save_json(folder/'source_hyperparameter_binding.json',{'at':now(),'selection_sha256':sha256(ROOT/settings['source_hyperparameter_selection']),
            'specification':source_spec,'source':'UCF source-validation AP only'})
        by_id={r['id']:r for r in read_json(ROOT/'data/ucf/manifest.json')['records'] if r['partition'] in {'train','validation'}}
        training=[by_id[i] for i in settings['source_training_ids']];validation=[by_id[i] for i in settings['source_validation_ids']]
        for variant in settings['systems']:
            spec={**source_spec,**variant,'reference':True}
            for seed in settings['seeds']:
                destination=folder/'runs'/f"{spec['name']}_seed{seed}"
                existed=(destination/'run.json').exists()
                if not existed and shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Less than3 GiB free before source-reference fit')
                save_json(folder/'status.json',{'state':'running','updated_at':now(),'pid':os.getpid(),
                    'completed_runs':completed,'planned_runs':9,'current':{'model':spec['name'],'seed':seed}})
                train(settings,spec,training,validation,seed,1.,destination);completed+=1;new+=int(not existed)
                if args.maximum_new_runs and new>=args.maximum_new_runs and completed<9:
                    save_json(folder/'status.json',{'state':'bounded_stop','updated_at':now(),'completed_runs':completed,'planned_runs':9});return
        save_json(folder/'status.json',{'state':'complete','updated_at':now(),'pid':os.getpid(),'completed_runs':completed,'planned_runs':9,
            'next':'Keep models frozen; no test scoring before one final confirmation registry'})
    except BaseException as error:
        save_json(folder/'status.json',{'state':'failed','updated_at':now(),'completed_runs':completed,'planned_runs':9,
            'error':str(error),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
