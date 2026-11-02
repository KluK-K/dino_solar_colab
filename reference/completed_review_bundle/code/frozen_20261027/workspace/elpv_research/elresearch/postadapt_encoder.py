"""Export source-trained LoRA into frozen official-architecture weights, with proof."""
import gc
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .common import ROOT,sha256,digest,read_json,save_json,now
from .adapt import inject_lora,merge_lora
from .encoders import FrozenEncoder,image_tensor
from .data import load_image


def save_merged_weights(model,folder,official_identity):
    """Lossless APFS clone + eight Q/V matrix writes; retain full HF-compatible files."""
    if not official_identity.get('model_id','').startswith('facebook/dinov3-'):
        model.to('cpu').save_pretrained(folder,safe_serialization=True)
        return {'method':'ordinary safetensors export; no registered official source clone'}
    from huggingface_hub import hf_hub_download
    source=Path(hf_hub_download(official_identity['model_id'],'model.safetensors',
        revision=official_identity['revision'],local_files_only=True))
    if sha256(source)!=official_identity['weights_sha256']:raise RuntimeError('Official export source changed')
    with source.open('rb') as stream:
        length=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(length))
    state=model.state_dict();keys=[f'layer.{i}.attention.{projection}.weight'
        for i in range(len(model.layer)-4,len(model.layer)) for projection in ['q_proj','v_proj']]
    for key in keys:
        if key not in header or header[key]['dtype']!='F32' or header[key]['shape']!=list(state[key].shape):
            raise RuntimeError('Official safetensors Q/V layout changed')
    temporary=folder/'merged.tmp.safetensors';temporary.unlink(missing_ok=True)
    try:
        subprocess.run(['/bin/cp','-c',str(source),str(temporary)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        method='APFS copy-on-write official safetensors clone with exact merged Q/V tensor updates'
    except subprocess.CalledProcessError:
        shutil.copyfile(source,temporary);method='ordinary official safetensors copy with exact merged Q/V tensor updates'
    written=0
    with temporary.open('r+b') as stream:
        for key in keys:
            value=state[key].detach().float().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes()
            begin,end=header[key]['data_offsets']
            if len(value)!=end-begin:raise RuntimeError('Merged projection byte size changed')
            stream.seek(8+length+begin);stream.write(value);written+=len(value)
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(folder/'model.safetensors');model.config.save_pretrained(folder)
    if sha256(source)!=official_identity['weights_sha256']:raise RuntimeError('Official source weights changed during clone export')
    return {'method':method,'updated_projection_tensors':len(keys),'updated_bytes':written,
        'logical_weights_bytes':source.stat().st_size,'official_source_weights_sha256':sha256(source),
        'verification':'Fresh official-architecture reload and token equality required before export metadata is committed'}


def export_source_encoder(parent_folder,training_rows,size,device,folder):
    folder.mkdir(parents=True,exist_ok=True)
    parent=read_json(parent_folder/'run.json')
    if parent['model']['adaptation']!='lora' or parent['model']['architecture']!='dinov3_multidepth':
        raise PermissionError('Only a source-trained, no-reference dense DINOv3 LoRA parent is eligible')
    if parent['model'].get('reference'):raise PermissionError('Parent must adapt without a stale normal bank')
    if sha256(parent_folder/'best.pt')!=parent['checkpoint_sha256']:raise RuntimeError('Parent checkpoint changed')
    if [r['id'] for r in training_rows]!=parent['training_ids']:
        raise PermissionError('Adaptation and subsequent heads must use exactly the same budgeted source images')
    signature={'parent_run_path':str((parent_folder/'run.json').relative_to(ROOT)),
        'parent_run_sha256':sha256(parent_folder/'run.json'),'parent_checkpoint_sha256':parent['checkpoint_sha256'],
        'source_training_ids':parent['training_ids'],'source_validation_ids':parent['validation_ids'],
        'parent_encoder':parent['encoder'],'input_size':size,'export_code_sha256':sha256(__file__),
        'merge_code_sha256':sha256(ROOT/'elresearch/adapt.py')}
    metadata=folder/'encoder_export.json'
    if metadata.exists():
        old=read_json(metadata)
        if old['signature']!=signature or old['weights_sha256']!=sha256(folder/'model.safetensors') or old['config_sha256']!=sha256(folder/'config.json'):
            raise RuntimeError('Frozen adapted-encoder export changed')
        return old
    encoder=FrozenEncoder('dinov3_small',size,str(device))
    if digest({k:v for k,v in encoder.identity.items() if k!='device'})!=digest(parent['encoder']):
        raise RuntimeError('Parent official encoder identity changed')
    torch.manual_seed(parent['seed']+10000);inject_lora(encoder.model,rank=8,blocks=4)
    state=torch.load(parent_folder/'best.pt',map_location='cpu',weights_only=True)
    adapters={name[len('encoder.'):]:value for name,value in state.items() if name.startswith('encoder.')}
    expected={name for name,p in encoder.model.named_parameters() if p.requires_grad}
    if set(adapters)!=expected or len(expected)!=16:raise RuntimeError('Missing/extra source LoRA parameters')
    result=encoder.model.load_state_dict(adapters,strict=False)
    if result.unexpected_keys:raise RuntimeError('Unexpected adapted parameters')
    # Check the source TRAINING image only; no target inference during export.
    image=image_tensor(load_image(training_rows[0]),size)[0][None].to(device)
    with torch.inference_mode():before=encoder.model(pixel_values=image).last_hidden_state.cpu().numpy()
    merge_lora(encoder.model)
    with torch.inference_mode():after=encoder.model(pixel_values=image).last_hidden_state.cpu().numpy()
    np.testing.assert_allclose(before,after,atol=3e-5,rtol=3e-5)
    storage=save_merged_weights(encoder.model,folder,signature['parent_encoder'])
    del encoder;gc.collect()
    if str(device)=='mps':torch.mps.empty_cache()
    restored=FrozenEncoder('dinov3_small',size,str(device),model_path=folder)
    with torch.inference_mode():again=restored.model(pixel_values=image).last_hidden_state.cpu().numpy()
    np.testing.assert_allclose(after,again,atol=2e-5,rtol=2e-5)
    if any(p.requires_grad for p in restored.model.parameters()):raise RuntimeError('Exported encoder must remain frozen')
    info={'at':now(),'signature':signature,'weights_sha256':sha256(folder/'model.safetensors'),
        'config_sha256':sha256(folder/'config.json'),'adapter_tensors':len(adapters),
        'merge_max_abs_error':float(np.max(abs(before-after))),'fresh_encoder_reload_max_abs_error':float(np.max(abs(after-again))),
        'source_training_probe_id':training_rows[0]['id'],
        'storage':storage,
        'normal_cache_policy':'Every feature/bank key binds these actual merged weights. No pre-adaptation normal cache may be reused.',
        'supervision':'Reuses exactly the parent source image/pixel budget; new stages add computation, not new annotation identities.'}
    save_json(metadata,info);del restored
    return info
