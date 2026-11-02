"""Annotation-blind full-pipeline timing; separate component instrumentation pass."""
import contextlib
import functools
import math
from pathlib import Path
import platform
import resource
import threading
import time

import numpy as np
import torch
from scipy.special import expit

from .common import now,digest
from .data import load_image
from .encoders import image_tensor,reproject


FORBIDDEN={'target','targets','mask','masks','mask_path','mask_sha256','y','native_mask'}


def source_image_rows(rows,permitted_ids):
    permitted=set(permitted_ids);output=[]
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Unique nonempty source image rows required')
    for row in rows:
        if row['id'] not in permitted or row['partition'] not in {'train','validation','target_development'} or row.get('dataset')=='elpv':
            raise PermissionError('Efficiency inputs must be audited observed research-source images, never sealed/course data')
        output.append({k:row[k] for k in ['id','dataset','partition','image_path','image_sha256','split_group']})
    return output


def synchronize(device):
    if torch.device(device).type=='mps':torch.mps.synchronize()
    elif torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def highwater_rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system()=='Darwin' else value*1024)


def source_decisions(logits,threshold,temperature):
    """Same strict score>threshold convention as metrics.classification."""
    logits=np.asarray(logits,np.float64)
    if not np.isfinite(logits).all() or not math.isfinite(threshold) or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid source operating point')
    return expit(logits/temperature),logits>threshold


def memory_snapshot(device):
    result={'process_lifetime_highwater_RSS_bytes':highwater_rss_bytes()}
    if torch.device(device).type=='mps':
        result.update(mps_tensor_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
    elif torch.device(device).type=='cuda':
        result.update(cuda_tensor_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    return result


class MemorySampler:
    """Only used during profiling, so its sampling does not affect primary latency."""
    def __init__(self,device,interval=.01):
        self.device,self.interval=device,interval;self.maximum={};self.samples=0
        self.stop=threading.Event();self.thread=None;self.errors=[]
    def sample(self):
        try:
            for key,value in memory_snapshot(self.device).items():self.maximum[key]=max(self.maximum.get(key,0),value)
            self.samples+=1
        except BaseException as error:self.errors.append(str(error));self.stop.set()
    def __enter__(self):
        self.sample()
        def loop():
            while not self.stop.wait(self.interval):self.sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.sample()
        if self.errors:raise RuntimeError('Memory observation failed: '+str(self.errors))
    def result(self):
        return {'maximum_observed':self.maximum,'samples':self.samples,'interval_seconds':self.interval,
            'scope':'Sampling in the separate component pass. MPS counters are observed allocations, not guaranteed peaks or CUDA VRAM. RSS highwater includes model loading/warmup. Unified-memory pools overlap; do not sum RSS and MPS counters.'}


class ComponentProfiler:
    def __init__(self,scorer):
        self.scorer,self.device=scorer,scorer.device;self.handles=[];self.patched=[];self.elapsed={};self.calls={}
    def reset(self):self.elapsed={};self.calls={}
    def add(self,name,seconds):
        self.elapsed[name]=self.elapsed.get(name,0.)+seconds;self.calls[name]=self.calls.get(name,0)+1
    def wrap(self,obj,name,category):
        old=getattr(obj,name);local=name in vars(obj);original=vars(obj).get(name)
        @functools.wraps(old)
        def timed(*args,**kwargs):
            synchronize(self.device);start=time.perf_counter()
            try:return old(*args,**kwargs)
            finally:synchronize(self.device);self.add(category,time.perf_counter()-start)
        setattr(obj,name,timed);self.patched.append((obj,name,local,original))
    def module(self,module,category,start_module=None,end_module=None):
        clock=[]
        def before(*args):synchronize(self.device);clock.append(time.perf_counter())
        def after(*args):
            synchronize(self.device)
            if not clock:raise RuntimeError('Unpaired component hook')
            self.add(category,time.perf_counter()-clock.pop())
        self.handles += [(start_module or module).register_forward_pre_hook(before),
                         (end_module or module).register_forward_hook(after)]
    def __enter__(self):
        scorer=self.scorer;model=scorer.model
        if scorer.binding['kind']=='classical':
            if scorer.encoder is None:
                from . import confirmation_predict
                self.wrap(confirmation_predict,'hog_statistics','feature_extraction')
            elif scorer.representation_pooling is not None:self.wrap(scorer.encoder,'encode','feature_extraction')
            else:self.module(scorer.encoder,'feature_extraction')
            self.wrap(model,'decision_function','classification')
        elif scorer.binding['kind']=='image':
            self.module(model.encoder,'feature_extraction');self.module(model.head,'classification')
        else:
            architecture=scorer.run['model']['architecture']
            if architecture=='published_psmt_deeplab':
                self.module(model.first.encoder,'feature_extraction')
                if hasattr(model,'second'):self.module(model.second.encoder,'feature_extraction')
            elif architecture=='deeplabv3plus_resnet50':
                self.module(model.encoder,'feature_extraction',model.encoder.conv1,model.encoder.layer4)
            else:self.module(model.encoder,'feature_extraction')
            if getattr(model,'bank',None) is not None:
                if architecture=='dinov3_normal_distance':
                    from . import mechanistic_models
                    self.wrap(mechanistic_models,'global_normal_distance','normal_reference_matching')
                else:self.wrap(model.bank,'residual','normal_reference_matching')
            if hasattr(model,'image_head'):self.module(model.image_head,'classification')
        return self
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        for obj,name,local,old in reversed(self.patched):
            if local:setattr(obj,name,old)
            else:delattr(obj,name)


def prepare_batch(rows,size):
    if any(FORBIDDEN&set(row) for row in rows):raise PermissionError('Annotation entered timing input')
    images=[];gray=[];geometry=[];valid=[];read_seconds=0.;preprocess_seconds=0.
    for row in rows:
        clock=time.perf_counter();image=load_image(row);read_seconds+=time.perf_counter()-clock
        clock=time.perf_counter();values,g=image_tensor(image,size)
        keep=torch.zeros(size,size,dtype=torch.bool);keep[g['top']:g['top']+g['h'],g['left']:g['left']+g['w']]=True
        images.append(values);gray.append(np.asarray(image,dtype=np.uint8));geometry.append(g);valid.append(keep)
        preprocess_seconds+=time.perf_counter()-clock
    clock=time.perf_counter()
    batch={'images':torch.stack(images),'gray_images':gray,'valid':torch.stack(valid),
        'geometries':geometry,'ids':[r['id'] for r in rows],'groups':[r['split_group'] for r in rows]}
    preprocess_seconds+=time.perf_counter()-clock
    return batch,{'read_checksum_decode':read_seconds,'preprocessing':preprocess_seconds}


def complete_batch(scorer,rows,threshold,temperature,profiler=None):
    synchronize(scorer.device);started=time.perf_counter()
    batch,stages=prepare_batch(rows,scorer.size)
    if profiler:profiler.reset()
    clock=time.perf_counter()
    output=scorer(batch);synchronize(scorer.device);stages['model_forward_and_transfers']=time.perf_counter()-clock
    clock=time.perf_counter();logits=np.asarray(torch.as_tensor(output['image_logits']).detach().cpu(),np.float64)
    if logits.shape!=(len(rows),) or not np.isfinite(logits).all():raise ValueError('Invalid timed image logits')
    probabilities,decisions=source_decisions(logits,threshold,temperature);native=[]
    maps=output.get('crack_probability')
    if maps is not None:
        maps=np.asarray(torch.as_tensor(maps).detach().cpu())
        if maps.ndim!=3 or maps.shape[0]!=len(rows) or not np.isfinite(maps).all() or maps.min()<-1e-6 or maps.max()>1+1e-6:
            raise ValueError('Timed crack outputs must be finite unscaled probabilities')
        native=[np.clip(reproject(np.clip(value,0,1),g),0,1) for value,g in zip(maps,batch['geometries'])]
    stages['decision_and_native_reconstruction']=time.perf_counter()-clock
    synchronize(scorer.device);total=time.perf_counter()-started
    # Numerical signature/bookkeeping intentionally outside primary batch timer.
    signatures=[]
    for i,row in enumerate(rows):
        samples=None
        if native:
            flat=native[i].ravel();index=np.linspace(0,len(flat)-1,min(512,len(flat))).round().astype(int)
            samples=flat[index].copy()
        signatures.append({'id':row['id'],'logit':float(logits[i]),'native_samples':samples})
    if profiler:
        stages.update({'component_'+k:v for k,v in profiler.elapsed.items()})
        remainder=stages['model_forward_and_transfers']-sum(profiler.elapsed.values())
        if remainder<-.0002:raise RuntimeError('Nested/overlapping component timings cannot be summed')
        stages['component_remaining_forward']=remainder
    return {'images':len(rows),'seconds':total,'stages':stages},signatures


def verify_signatures(reference,current):
    if reference.keys()!=current.keys():raise RuntimeError('Timing pass image identities differ')
    maximum_logit=0.;maximum_pixel=0.;logit_ratio=0.;pixel_ratio=0.
    for key in reference:
        a,b=reference[key],current[key]
        np.testing.assert_allclose(b['logit'],a['logit'],atol=2e-5,rtol=2e-5)
        maximum_logit=max(maximum_logit,abs(a['logit']-b['logit']))
        logit_ratio=max(logit_ratio,abs(a['logit']-b['logit'])/(2e-5+2e-5*abs(a['logit'])))
        if (a['native_samples'] is None)!=(b['native_samples'] is None):raise RuntimeError('Timing pass lost localization outputs')
        if a['native_samples'] is not None:
            np.testing.assert_allclose(b['native_samples'],a['native_samples'],atol=2e-5,rtol=2e-5)
            maximum_pixel=max(maximum_pixel,float(np.max(abs(a['native_samples']-b['native_samples']))))
            pixel_ratio=max(pixel_ratio,float(np.max(abs(a['native_samples']-b['native_samples'])/(2e-5+2e-5*abs(a['native_samples'])))))
    return {'logit_max_absolute_error':maximum_logit,'sampled_native_probability_max_absolute_error':maximum_pixel,
        'maximum_logit_scaled_tolerance_ratio':logit_ratio,'maximum_pixel_scaled_tolerance_ratio':pixel_ratio,
        'scope':'All image logits plus up to512 fixed uniform native pixels/image, not every native probability retained'}


def pass_summary(batches,loop_seconds):
    n=sum(b['images'] for b in batches);total=sum(b['seconds'] for b in batches)
    per_image=np.array([b['seconds']/b['images'] for b in batches])
    names=set().union(*(b['stages'] for b in batches))
    return {'images':n,'batches':len(batches),'timed_pipeline_seconds':total,'observed_loop_seconds':loop_seconds,
        'weighted_mean_ms_per_image':total/n*1000,'pipeline_images_per_second':n/total,
        'observed_loop_images_per_second':n/loop_seconds,
        'batch_normalized_ms_per_image_percentile':{str(q):float(np.percentile(per_image,q)*1000) for q in [50,90,95]},
        'stage_sum_seconds':{name:sum(b['stages'].get(name,0.) for b in batches) for name in sorted(names)},
        'batch_normalized_percentile_limit':'Distribution over batches, not independently measured per-image latency at batch>1. Partial final batches use their actual image counts.',
        'timing_limit':'Primary pipeline sum excludes output signatures, profiling hooks/sampling, model loading and output-file serialization; observed loop time includes signature/bookkeeping overhead.'}


def measure_cohort(scorer,rows,settings,threshold,temperature,cohort_name,signature_path=None):
    if not math.isfinite(threshold) or not math.isfinite(temperature) or temperature<=0:raise ValueError('Source operating point invalid')
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Unique nonempty timing cohort required')
    reference=None;results=[];canonical=sorted(rows,key=lambda r:r['id'])
    if any(FORBIDDEN&set(row) for row in rows):raise PermissionError('Annotations must be projected out before profiling')
    for batch_size in settings['batch_sizes']:
        if batch_size<1:raise ValueError('Positive batch sizes required')
        warm=[canonical[i%len(canonical)] for i in range(batch_size)]
        for _ in range(settings['warmup_batches']):complete_batch(scorer,warm,threshold,temperature)
        if len(rows)%batch_size:complete_batch(scorer,canonical[-(len(rows)%batch_size):],threshold,temperature)
        repeats=[];proof=[];raw_records=[]
        for repetition in range(settings['repetitions']):
            rng=np.random.default_rng(np.random.SeedSequence([settings['seed'],repetition,239]))
            order=rng.permutation(len(canonical));ordered=[canonical[i] for i in order]
            batches=[];signatures={};clock=time.perf_counter()
            for start in range(0,len(ordered),batch_size):
                timing,values=complete_batch(scorer,ordered[start:start+batch_size],threshold,temperature)
                batches.append(timing);signatures.update({v['id']:v for v in values})
            repeats.append(pass_summary(batches,time.perf_counter()-clock));raw_records.append(batches)
            if reference is None:reference=signatures
            proof.append(verify_signatures(reference,signatures))
        # Timings with synchronized hooks are separate and cannot replace latency.
        component_records=[];signatures={};clock=time.perf_counter()
        with MemorySampler(scorer.device,settings['memory_sampling_interval']) as memory,ComponentProfiler(scorer) as profiler:
            for start in range(0,len(canonical),batch_size):
                timing,values=complete_batch(scorer,canonical[start:start+batch_size],threshold,temperature,profiler)
                component_records.append(timing);signatures.update({v['id']:v for v in values})
        proof.append(verify_signatures(reference,signatures))
        results.append({'cohort':cohort_name,'batch_size':batch_size,'input_size':scorer.size,
            'repeated_uninstrumented_passes':repeats,'raw_batch_timings':raw_records,
            'component_pass':pass_summary(component_records,time.perf_counter()-clock),
            'component_interpretation':'Separate synchronized instrumentation pass; remaining forward includes decoder, pooling, uninstrumented normalization/transfers and synchronization overhead. Feature-stage boundaries differ across architectures.',
            'memory':memory.result(),'prediction_equivalence':proof})
    if signature_path is not None:
        path=Path(signature_path);path.parent.mkdir(parents=True,exist_ok=True)
        ordered=[reference[r['id']] for r in canonical];samples=[r['native_samples'] for r in ordered]
        lengths=[len(v) if v is not None else 0 for v in samples]
        np.savez_compressed(path,ids=np.array([r['id'] for r in canonical]),
            logits=np.array([r['logit'] for r in ordered]),native_sample_offsets=np.cumsum([0]+lengths),
            native_sample_values=np.concatenate([v for v in samples if v is not None]) if any(lengths) else np.empty(0,np.float32))
    return results
