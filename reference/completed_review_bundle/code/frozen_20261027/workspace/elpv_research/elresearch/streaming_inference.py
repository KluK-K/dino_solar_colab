"""Bounded-memory image inference; public development entry point never opens sealed partitions."""
import time

import numpy as np
import torch

from .data import load_image
from .encoders import image_tensor,reproject


def image_batches(rows,size,batch_size):
    if batch_size<1:raise ValueError('Positive batch size required')
    for start in range(0,len(rows),batch_size):
        selected=rows[start:start+batch_size];prepared=[];gray_images=[];valid=[];geometries=[]
        for row in selected:
            image=load_image(row);gray=np.asarray(image,dtype=np.uint8)
            tensor,g=image_tensor(image,size);keep=torch.zeros(size,size,dtype=torch.bool)
            keep[g['top']:g['top']+g['h'],g['left']:g['left']+g['w']]=True
            prepared.append(tensor);gray_images.append(gray);valid.append(keep);geometries.append(g)
        yield selected,{'images':torch.stack(prepared),'gray_images':gray_images,
            'valid':torch.stack(valid),'geometries':geometries,'ids':[r['id'] for r in selected],
            'groups':[r.get('split_group') for r in selected]}


def _stream_image_scores(rows,scorer,size,batch_size,consume_map=None):
    """Internal numerical engine. A separate registry gate must authorize any confirmation caller."""
    identities=[];scores=[];maximum=0;started=time.perf_counter();map_count=0
    with torch.inference_mode():
        for selected,batch in image_batches(rows,size,batch_size):
            # No targets or annotation masks are supplied to the scorer.
            result=scorer(batch)
            logits=torch.as_tensor(result['image_logits']).detach().cpu().numpy().astype(np.float64)
            if logits.shape!=(len(selected),) or not np.isfinite(logits).all():raise ValueError('Invalid streamed image logits')
            identities.extend(row['id'] for row in selected);scores.extend(logits.tolist())
            maximum=max(maximum,len(selected));maps=result.get('crack_probability')
            if maps is not None:
                maps=torch.as_tensor(maps).detach().cpu().numpy()
                if maps.ndim!=3 or maps.shape[0]!=len(selected) or not np.isfinite(maps).all():raise ValueError('Invalid streamed crack maps')
                if maps.min()<-1e-6 or maps.max()>1+1e-6:raise ValueError('Crack maps must be probabilities, not per-image rescaled activations')
                if consume_map is None:raise ValueError('A map consumer is required; maps may not silently accumulate')
                for row,logit,probability,g in zip(selected,logits,maps,batch['geometries']):
                    native=np.clip(reproject(np.clip(probability,0,1),g),0,1)
                    consume_map(row,float(logit),native);map_count+=1
            del batch,result
    elapsed=time.perf_counter()-started
    return {'ids':np.array(identities),'logits':np.asarray(scores,dtype=np.float64),
        'native_maps_consumed':map_count,'maximum_prepared_images':maximum,
        'elapsed_decode_preprocess_inference_seconds':elapsed,
        'timing_limit':'includes checksum reads, decoding, preprocessing and first model call; concurrent local work and warm-up can affect timing; not an isolated inference benchmark',
        'map_retention':'maps delivered one at a time to a consumer; this engine does not retain a cohort-sized map array'}


def stream_development_scores(rows,scorer,size,batch_size,permitted_ids,consume_map=None):
    rows=list(rows);permitted_ids=set(permitted_ids)
    if not rows:raise ValueError('No development rows')
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Repeated prediction identities')
    if any(r['id'] not in permitted_ids or r['partition'] not in {'train','validation','target_development'} for r in rows):
        raise PermissionError('Sealed or unregistered images require a separate frozen confirmation registry')
    if any(r.get('dataset')=='elpv' for r in rows):raise PermissionError('The ELPV course workflow is separate')
    return _stream_image_scores(rows,scorer,size,batch_size,consume_map)
