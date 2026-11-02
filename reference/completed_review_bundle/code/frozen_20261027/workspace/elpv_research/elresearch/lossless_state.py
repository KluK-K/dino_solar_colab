"""Lossless XOR/byte-shuffled checkpoint storage; all parameters and BN buffers retained."""
import gzip
import hashlib
import io
from pathlib import Path
import os

import numpy as np
import torch


def state_digest(state):
    digest=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous();array=value.numpy()
        digest.update(key.encode());digest.update(str(array.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def save_lossless_state(path,state,basis,basis_binding):
    if set(state)!=set(basis):raise ValueError('Checkpoint/basis keys differ')
    values={};total=0
    for key,value in state.items():
        current=value.detach().cpu().contiguous().numpy();initial=basis[key].detach().cpu().contiguous().numpy()
        if current.shape!=initial.shape or current.dtype!=initial.dtype:raise ValueError('Checkpoint/basis tensor schema differs: '+key)
        if np.issubdtype(current.dtype,np.floating) and not np.isfinite(current).all():raise ValueError('Nonfinite checkpoint tensor')
        raw=current.reshape(-1).view(np.uint8).reshape(-1,current.dtype.itemsize)
        base=initial.reshape(-1).view(np.uint8).reshape(raw.shape)
        # Shuffle byte lanes before compression; restore reverses this operation exactly.
        shuffled=np.bitwise_xor(raw,base).T.copy()
        values[key]=torch.from_numpy(shuffled);total+=current.nbytes
    record={'format':'xor_byteshuffle_gzip_v1','basis_binding':basis_binding,'basis_tensor_sha256':state_digest(basis),
        'state_tensor_sha256':state_digest(state),'xor_byte_lanes':values,'uncompressed_tensor_bytes':total}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with gzip.open(temporary,'wb',compresslevel=3) as stream:torch.save(record,stream)
    temporary.replace(path)
    return {'format':record['format'],'basis_tensor_sha256':record['basis_tensor_sha256'],
        'state_tensor_sha256':record['state_tensor_sha256'],'uncompressed_tensor_bytes':total,
        'checkpoint_bytes':path.stat().st_size,'compression_ratio':path.stat().st_size/total}


def load_lossless_state(path,basis,basis_binding):
    with gzip.open(path,'rb') as stream:record=torch.load(io.BytesIO(stream.read()),map_location='cpu',weights_only=True)
    if record['format']!='xor_byteshuffle_gzip_v1' or record['basis_binding']!=basis_binding:
        raise RuntimeError('Lossless checkpoint format/basis binding changed')
    if state_digest(basis)!=record['basis_tensor_sha256'] or set(basis)!=set(record['xor_byte_lanes']):
        raise RuntimeError('Exact checkpoint basis tensors changed')
    state={}
    for key,initial in basis.items():
        original=initial.detach().cpu().contiguous().numpy()
        raw=original.reshape(-1).view(np.uint8).reshape(-1,original.dtype.itemsize)
        delta=record['xor_byte_lanes'][key].numpy().T
        if delta.shape!=raw.shape or delta.dtype!=np.uint8:raise RuntimeError('Invalid checkpoint byte schema')
        restored=np.bitwise_xor(raw,delta).copy().reshape(-1).view(original.dtype).reshape(original.shape)
        state[key]=torch.from_numpy(restored.copy())
    if state_digest(state)!=record['state_tensor_sha256']:raise RuntimeError('Lossless checkpoint reconstruction changed')
    return state,record
