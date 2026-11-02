"""Full-source inference benchmark after all registered model development completes."""
import argparse
import fcntl
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from elresearch.common import ROOT,read_json,save_json,sha256,digest,now
from elresearch.confirmation_registry import EVALUATOR_MODULES,bind_model,bind_file,check_file,runtime_versions
from elresearch.inference_efficiency import source_image_rows,measure_cohort,synchronize,memory_snapshot
from elresearch.confirmation_predict import FrozenScorer
from elresearch.efficiency_evidence import verify_completed_benchmark

BASE=ROOT/'artifacts/publication_20261017'
STUDIES={'image':'scdd_development_v1','classical':'classical_controls_v1','dense':'dense_development_v1',
 'ucf_reference_transfer':'ucf_reference_transfer_v1','cnn_stability':'cnn_stability_v1',
 'representation_scale':'representation_scale_v1','mechanistic':'mechanistic_controls_v1','published_methods':'published_methods_v1'}
FILES=list(dict.fromkeys(EVALUATOR_MODULES+['inference_efficiency.py','efficiency_evidence.py']))


def model_schedule():
    jobs=[]
    def add(kind,family,study,name,budget,seeds,view='primary'):
        for seed in seeds:
            jobs.append({'kind':kind,'family':family,'seed':seed,'study':study,'source_view':view,
                'run_folder':f'artifacts/publication_20261017/{study}/runs/internal/budget_{budget:g}/{name}_seed{seed}'})
    for name in ['dinov3_frozen','dinov3_lora','dinov3_full','resnet18_full','resnet50_full','efficientnet_b0_full']:
        add('image','image__'+name,STUDIES['image'],name,1.,[11,22,33])
    for feature in ['hog_stats','dinov3_cls']:
        for method in ['logistic','linear_svm','rbf_svm']:
            name=feature+'_'+method;add('classical','classical__'+name,STUDIES['classical'],name,1.,[11])
    representation=read_json(BASE/STUDIES['representation_scale']/'protocol.json')
    for name in dict.fromkeys(s['family'] for s in representation['grid']):
        add('classical','representation__'+name,STUDIES['representation_scale'],name,1.,[11])
    for name in ['deeplabv3plus_full','dinov3_dense_frozen','dinov3_dense_lora','dinov3_dense_full',
        'dinov3_dense_reference','dinov3_reference_no_position','dinov3_reference_mean_evidence']:
        add('dense','dense__'+name,STUDIES['dense'],name,1.,[11,22,33])
    for name in ['dinov3_patch_mlp','dinov3_postlora_no_reference','dinov3_postlora_reference']:
        add('dense','mechanism__'+name,STUDIES['mechanistic'],name,1.,[11,22,33])
    add('dense','mechanism__dinov3_global_normal_distance',STUDIES['mechanistic'],'dinov3_global_normal_distance',1.,[11])
    for budget in [.2,1.]:
        add('dense',f'published__supervised_b{int(budget*100)}',STUDIES['published_methods'],'published_deeplab_fsl',budget,[11,22,33])
    for view in ['primary','ensemble']:
        add('dense','published__ssl_'+view,STUDIES['published_methods'],'published_psmt_ssl',.2,[11,22,33],view)
    # Normal memory changes with annotation budget, so reusing its full-budget
    # latency for20% would conceal a real deployment-cost difference.
    for name in ['dinov3_dense_reference','dinov3_reference_no_position','dinov3_reference_mean_evidence']:
        add('dense','dense__'+name+'_b20',STUDIES['dense'],name,.2,[11,22,33])
    for name in ['dinov3_postlora_reference','dinov3_global_normal_distance']:
        add('dense','mechanism__'+name+'_b20',STUDIES['mechanistic'],name,.2,[11,22,33])
    if len(jobs)!=97 or len({(j['family'],j['seed']) for j in jobs})!=97:
        raise RuntimeError('Efficiency schedule must cover all97 predeclared systems/seed/views')
    return jobs


def scientific_hashes():return {name:sha256(ROOT/'elresearch'/name) for name in FILES}


def host_identity():
    value={'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor()}
    if platform.system()=='Darwin':
        for name,key in [('cpu_brand','machdep.cpu.brand_string'),('physical_memory_bytes','hw.memsize'),('physical_cpu_cores','hw.physicalcpu')]:
            raw=subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip()
            value[name]=raw if name=='cpu_brand' else int(raw)
    return value


def design():
    cohorts={};manifests=[]
    for name in ['scdd_publication_v1','ucf']:
        path=ROOT/'data'/name/'manifest.json';rows=[r for r in read_json(path)['records'] if r['partition']=='train']
        cohorts[name]=source_image_rows(sorted(rows,key=lambda r:r['id']),[r['id'] for r in rows])
        manifests.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
    if len(cohorts['scdd_publication_v1'])!=351 or len(cohorts['ucf'])!=430:raise RuntimeError('Benchmark source cohort changed')
    value={'version':2,'kind':'Complete source-only end-to-end inference efficiency',
        'amendment':{'parent_protocol':{'path':'artifacts/publication_20261017/inference_efficiency_v1/protocol_revisions/de2657c61014eeb9f77a/protocol.json',
            'sha256':'013608ec388b64d841fbf4f3b9a8431f429f1b6cdcbfc5bcc9f0e8133ffe1085'},
            'reason':'Before any formal timing, align efficiency decision ties with strict score>source_threshold used by source calibration/classification. Scores, models, masks, cohorts and selection do not change.',
            'formal_jobs_completed_before_amendment':0},
        'registered_after':'Source image development and published-method implementation; before any formal publication-system benchmark or final confirmation',
        'planned_jobs':97,'models':model_schedule(),'cohorts':cohorts,'source_manifests':manifests,
        'upstream_protocols':{name:{'path':str((BASE/folder/'protocol.json').relative_to(ROOT)),
            'sha256':sha256(BASE/folder/'protocol.json')} for name,folder in STUDIES.items()},
        'batch_sizes':[1,8],'warmup_batches':5,'repetitions':3,'seed':20261017,'memory_sampling_interval':.01,
        'minimum_free_GiB_before_job':1.,
        'device':'mps','torch_threads':4,'runtime_versions':runtime_versions(),'host':host_identity(),
        'source_scope':'Exactly351 SCDD+430 UCF already-observed source-training images, no labels/masks supplied to timing/scorer. These are compute inputs, not new model-performance observations.',
        'coverage':'97 internal-source model/seed/view cases:82 cases spanning6 neural image systems,6 classical probes,15 pooling/resolution/scale/CNN probes,7 dense controls,patch/distance/post-LoRA and complete published FSL/SSL;15 additional20%-budget reference/distance cases explicitly measure annotation-dependent memory costs. Representative fitted systems, not latency of every fold/budget checkpoint.',
        'timing':'Primary repeated passes have no hooks/sampler. Read/checksum/decode, preprocessing, actual model/normal search/transfers, source decision and native-map reconstruction are included. Model initialization/load measured separately. Warm filesystem/model, no OS cache flush or deployment claim.',
        'components':'Additional full-corpus synchronized-hook pass; numerical agreement checked against uninstrumented predictions. Components do not replace primary latency and feature-boundary definitions differ by architecture.',
        'memory':'Separate10ms sampler during component pass; MPS observed tensor/driver maxima are lower-bound observations, not guaranteed peaks/CUDA VRAM. Separate child per model; RSS lifetime highwater includes model load. Unified-memory pools are not disjoint.',
        'label_and_model_selection':'Frozen source-selected runs/operating points; no fitting/accuracy selection in benchmark. Full budgets, published FSL/SSL20% and all reference/distance methods at20%; report annotation/reference counts. Ordinary weight-value changes do not create a distinct architecture benchmark.',
        'repetition_limit':'Three source-trained neural seeds retained; deterministic full-budget probes saved once. Timing repeats characterize local run variation, not independent accuracy experiments.',
        'limitations':['Actual M2 Max32GB host; no claim of unavailable RTX4070/500GB-server measurement.',
            'Other desktop activity may vary; own registered training/probe workers must all finish/release locks before formal timing.',
            'Reference query-family exclusion and actual bank size remain active; report source/input population and bank tensors/serialized bytes.',
            'HOG and representation probes use the same real frozen-scorer preprocessing path, including any redundant generic preparation; not a claim of an optimized deployment.',
            'Output file serialization, signatures, plotting and optimizer training are excluded from primary batch timings; total loop and loading reported separately.'],
        'scientific_hashes':scientific_hashes(),'runner_sha256':sha256(__file__)}
    value['study_id']=digest(value)[:20];return value


def register(folder):
    proposal=design();path=folder/'protocol.json'
    if path.exists():
        old=read_json(path)
        if digest({k:v for k,v in old.items() if k!='registered_at'})!=digest(proposal):raise RuntimeError('Registered full-inference protocol changed')
        return old
    snapshot=folder/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for filename in FILES:shutil.copy2(ROOT/'elresearch'/filename,snapshot/filename)
    shutil.copy2(__file__,snapshot/Path(__file__).name)
    proposal['registered_at']=now();save_json(path,proposal);return proposal


def check_binding(settings):
    if scientific_hashes()!=settings['scientific_hashes'] or sha256(__file__)!=settings['runner_sha256']:
        raise RuntimeError('Registered efficiency implementation changed')
    if runtime_versions()!=settings['runtime_versions']:raise RuntimeError('Registered timing runtime changed')
    if host_identity()!=settings['host']:raise RuntimeError('Registered inference measurement host changed')
    for binding in settings['source_manifests']+list(settings['upstream_protocols'].values()):check_file(binding)
    check_file(settings['amendment']['parent_protocol'])


def normal_memory_assets(binding,scorer):
    recipe=binding.get('normal_bank_recipe');paths=[]
    if recipe:
        r=read_json(check_file(recipe))
        paths=([ROOT/r['payload']['path']] if r.get('format')=='canonical_orientation_memory_v2' else
            [ROOT/r['cache_directory']/(row['cache_key']+'.pt') for row in r['rows']])
    bank=getattr(scorer.model,'bank',None);seen=set()
    def count(value):
        if isinstance(value,torch.Tensor):
            storage=value.untyped_storage();key=(str(value.device),storage.data_ptr())
            if not value.numel() or key in seen:return 0
            seen.add(key);return storage.nbytes()
        if isinstance(value,dict):return sum(count(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(count(v) for v in value)
        return 0
    return {'normal_reference_images':len(bank.items) if bank else 0,
        'normal_reference_tensor_storage_bytes':count(vars(bank)) if bank else 0,
        'normal_reference_serialized_bytes':sum(path.stat().st_size for path in set(paths)),
        'tensor_storage_scope':'Actual unique loaded tensor storages, including normalization/retrieval buffers, excluding Python object overhead.'}


def worker(folder,job_id):
    settings=read_json(folder/'protocol.json');check_binding(settings)
    jobs=read_json(folder/'model_bindings.json');job=next(j for j in jobs['jobs'] if j['id']==job_id)
    if jobs['protocol_sha256']!=sha256(folder/'protocol.json'):raise RuntimeError('Frozen benchmark model list changed')
    binding=job['binding'];destination=folder/'jobs'/job_id;destination.mkdir(parents=True,exist_ok=True)
    if (destination/'result.json').exists():return
    canonical=bind_model(binding['kind'],binding['family'],binding['seed'],check_file(binding['run']).parent,
        check_file(binding['source_protocol']),binding['source_view'])
    if canonical!=binding:raise RuntimeError('Benchmark source model/operating point changed')
    torch.set_num_threads(settings['torch_threads']);torch.manual_seed(settings['seed'])
    if settings['device']=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Registered MPS host unavailable')
    clock=time.perf_counter();scorer=FrozenScorer(binding,settings['device']);synchronize(scorer.device)
    loading=time.perf_counter()-clock
    if scorer.device.type=='mps':torch.mps.empty_cache()
    modules=[]
    if isinstance(scorer.model,torch.nn.Module):modules.append(scorer.model)
    if scorer.encoder is not None:modules.append(scorer.encoder if isinstance(scorer.encoder,torch.nn.Module) else scorer.encoder.model)
    parameters={id(p):p for module in modules for p in module.parameters()}
    result={'at':now(),'study_id':settings['study_id'],'model_id':job_id,
        'protocol_sha256':sha256(folder/'protocol.json'),'model_bindings_sha256':sha256(folder/'model_bindings.json'),
        'binding':binding,'host':{**host_identity(),'device':str(scorer.device),'torch':torch.__version__},
        'cold_model_and_bank_load_seconds':loading,'cold_load_scope':'Source implementation/checkpoint verification, architecture/init, full state restoration, normal bank load and device transfer; process/import startup excluded; filesystem may be warm.',
        'checkpoint_bytes':check_file(binding['checkpoint']).stat().st_size,
        'loaded_torch_parameters':sum(p.numel() for p in parameters.values()),
        'loaded_torch_parameter_bytes':sum(p.numel()*p.element_size() for p in parameters.values()),
        'parameter_count_scope':'PyTorch tensors only; classical estimators also contain learned NumPy coefficients/support vectors represented in checkpoint_bytes, not a zero-parameter method.',
        'model_source_training_images':len(binding['source_train_ids']),
        'model_source_validation_images':len(binding['source_validation_ids']),
        'model_source_unlabelled_images':len(binding.get('source_unlabelled_ids',[])),
        **normal_memory_assets(binding,scorer),'cohorts':{},'numerical_signature_files':{}}
    for name,rows in settings['cohorts'].items():
        signature_path=destination/(name+'_numerical_signatures.npz')
        result['cohorts'][name]=measure_cohort(scorer,rows,settings,binding['source_threshold'],binding['source_temperature'],name,signature_path)
        result['numerical_signature_files'][name]=bind_file(signature_path)
        save_json(destination/'progress.json',{'at':now(),'state':'running','completed_cohorts':list(result['cohorts'])})
    result['memory_after_benchmark']=memory_snapshot(scorer.device)
    result['prediction_equivalence_verified']=True
    save_json(destination/'result.json',result)
    print('BENCHMARK COMPLETE',job_id,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--wait-for-development',action='store_true');parser.add_argument('--job-id');args=parser.parse_args()
    folder=BASE/'inference_efficiency_v1';folder.mkdir(exist_ok=True)
    if args.job_id:return worker(folder,args.job_id)
    lock=(folder/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=register(folder);print('FULL INFERENCE REGISTERED',settings['study_id'],settings['planned_jobs'],'model/seed/views',flush=True)
    if args.register_only:return
    completed=0
    def status(state,**extra):save_json(folder/'status.json',{'at':now(),'state':state,'pid':os.getpid(),'completed_jobs':completed,'planned_jobs':settings['planned_jobs'],**extra})
    try:
        while True:
            states={key:read_json(BASE/value/'status.json')['state'] for key,value in STUDIES.items()}
            if all(s=='complete' for s in states.values()):break
            if not args.wait_for_development or any(s in {'failed','bounded_stop'} for s in states.values()):
                raise RuntimeError('Model-development studies need completion/resume before isolated timing: '+str(states))
            status('queued_after_all_development',upstream_states=states);time.sleep(30)
        check_binding(settings)
        # Parent status may precede process lock release. Hold every source lock
        # plus the shared GPU lock throughout formal isolated measurements.
        locks=[]
        for name in STUDIES.values():
            handle=(BASE/name/'runner.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX);locks.append(handle)
        gpu=(BASE/'gpu_work_queue.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX)
        jobs=[]
        for spec in settings['models']:
            protocol=BASE/spec['study']/'protocol.json'
            binding=bind_model(spec['kind'],spec['family'],spec['seed'],ROOT/spec['run_folder'],protocol,spec['source_view'])
            jobs.append({'id':binding['id'],'binding':binding})
        random.Random(settings['seed']).shuffle(jobs)
        model_bindings={'protocol_sha256':sha256(folder/'protocol.json'),'jobs':jobs}
        path=folder/'model_bindings.json'
        if path.exists() and read_json(path)!=model_bindings:raise RuntimeError('Predeclared benchmark model list changed')
        save_json(path,model_bindings)
        for job in jobs:
            check_binding(settings);status('running',current=job['id'])
            if shutil.disk_usage(ROOT).free<settings['minimum_free_GiB_before_job']*1024**3:
                raise RuntimeError('Insufficient storage for benchmark records; preserve all fitted models and resolve storage')
            subprocess.run([str((ROOT/'.venv/bin/python').absolute()),'-u',str(Path(__file__).absolute()),'--job-id',job['id']],
                cwd=ROOT,check=True)
            completed+=1
        if completed!=settings['planned_jobs']:raise RuntimeError('Full benchmark model count differs')
        save_json(folder/'verification.json',verify_completed_benchmark(folder,require_complete_status=False))
        status('complete',interpretation='Source-only full-pipeline efficiency, no final confirmation scored; keep GPU/RSS and timing limitations explicit')
    except BaseException as error:
        status('failed',error=str(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
