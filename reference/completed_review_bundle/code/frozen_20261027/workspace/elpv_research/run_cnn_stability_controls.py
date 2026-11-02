"""Source-only paired LR sensitivity; never discard the original weaker CNN runs."""
import argparse
import fcntl
import os
import shutil
import time
import traceback

from elresearch.common import ROOT,read_json,save_json,digest,sha256,now
from elresearch.publication_splits import fold_records
from elresearch.publication_train import FILES,scientific_hashes,train
from verify_publication_runs import verify


BASE=ROOT/'artifacts/publication_20261017'
PARENT=BASE/'ucf_reference_transfer_v1'


def register(folder):
    original=read_json(BASE/'scdd_development_v1/protocol.json')
    specifications=[]
    for spec in original['specifications']:
        if spec['encoder']=='dinov3_small':continue
        for rate in [1e-5,3e-4]:
            specifications.append({**spec,'name':spec['name']+f'_lr{rate:g}','encoder_lr':rate})
    fold=next(f for f in original['split_manifest']['folds'] if f['name']=='internal')
    settings={key:original[key] for key in ['input_size','batch_size','updates_per_round','maximum_rounds','patience',
              'weight_decay','target_source_recall','device','calibration_protocol_sha256','split_manifest']}
    settings.update(version=1,kind='CNN source-only optimization stability diagnostic',
        rationale='One full-label EfficientNet seed had source AUROC0.806 and training-step BCE0.645. Check source fitting and paired LR sensitivity before interpreting DINO/CNN differences.',
        registered_after='All72 internal image-only source results; before any independent confirmation',
        specifications=specifications,seeds=[11,22,33],planned_runs=18,
        source_training_ids=fold['source_train_ids'],source_validation_ids=fold['source_validation_ids'],
        scientific_hashes=scientific_hashes(),runner_sha256=sha256(__file__),
        source_verifier_sha256=sha256(ROOT/'verify_publication_runs.py'),
        original_protocol_sha256=sha256(BASE/'scdd_development_v1/protocol.json'),
        changes='Only encoder LR changes from1e-4 to1e-5 or3e-4. Same heads/init seeds, D4 batches, fixed maximum updates, BN policy, early stopping and source labels.',
        scope='Post-hoc source DEVELOPMENT diagnostic at100% training labels. Retain original results; no test data choose rates. Do not use these full-label fits as5%-label experiments.',
        pending_after='If model fitting remains weak, distinguish normalization/head initialization and optimization before a universal-backbone advantage claim.')
    settings['study_id']=digest(settings)[:20];path=folder/'protocol.json'
    if path.exists():
        old=read_json(path)
        if digest({k:v for k,v in old.items() if k!='registered_at'})!=digest(settings):raise RuntimeError('Registered source CNN stability controls changed')
        return old
    snapshot=folder/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for name in FILES:shutil.copy2(ROOT/'elresearch'/name,snapshot/name)
    shutil.copy2(__file__,snapshot/'run_cnn_stability_controls.py')
    settings['registered_at']=now();save_json(path,settings);return settings


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--wait-for-ucf-reference',action='store_true');args=parser.parse_args()
    folder=BASE/'cnn_stability_v1';folder.mkdir(exist_ok=True)
    lock=(folder/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=register(folder);print('CNN STABILITY REGISTERED',settings['study_id'],'18 source-only paired fits',flush=True)
    if args.register_only:return
    completed=0
    try:
        while True:
            status=read_json(PARENT/'status.json')
            if status['state']=='complete':
                with (PARENT/'runner.lock').open('a') as prior:fcntl.flock(prior,fcntl.LOCK_EX)
                break
            if not args.wait_for_ucf_reference or status['state'] in {'failed','bounded_stop'}:raise RuntimeError('Prior source-reference work needs completion/resume')
            save_json(folder/'status.json',{'state':'queued_after_ucf_reference','at':now(),'pid':os.getpid(),'completed_runs':0,'planned_runs':18,'parent_state':status['state']})
            time.sleep(30)
        gpu_lock=(BASE/'gpu_work_queue.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
        training,validation,_=fold_records(settings['split_manifest'],'internal')
        for spec in settings['specifications']:
            for seed in settings['seeds']:
                destination=folder/'runs'/f"{spec['name']}_seed{seed}"
                if not (destination/'run.json').exists() and shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Less than3 GiB free before CNN control')
                save_json(folder/'status.json',{'state':'running','at':now(),'pid':os.getpid(),'completed_runs':completed,'planned_runs':18,'current':{'model':spec['name'],'seed':seed}})
                train(settings,spec,training,validation,seed,1.,destination)
                verify(destination,validation,settings['input_size'],settings['batch_size']);completed+=1
        save_json(folder/'status.json',{'state':'complete','at':now(),'pid':os.getpid(),'completed_runs':completed,'planned_runs':18,
            'next':'Compare source fitting and held-source evidence; no automatic confirmation'})
    except BaseException as error:
        save_json(folder/'status.json',{'state':'failed','at':now(),'completed_runs':completed,'planned_runs':18,
            'error':str(error),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
