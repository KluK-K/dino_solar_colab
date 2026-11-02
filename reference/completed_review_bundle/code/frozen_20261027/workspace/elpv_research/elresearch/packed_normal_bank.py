"""Persist exact inference references; regenerate hashed D4 training intermediates.

The canonical view0 tensors are immutable inference artifacts. Views1..7 are
recoverable training caches: their tensor hashes, source images/masks, official
weights, merged source adaptation and transform code remain registered.
"""
import gzip
import hashlib
import io
from pathlib import Path

import torch
from torch.nn import functional as F

from .common import ROOT,digest,sha256,read_json,save_json,now
from .publication_normal_bank import OrientationNormalBank


def view_digest(view):
    signature={}
    for key in ['cls','patch','coords']:
        tensor=view[key].detach().cpu().contiguous()
        signature[key]={'shape':list(tensor.shape),'dtype':str(tensor.dtype),
                        'sha256':hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
    return digest(signature)


class PackedNormalBank(OrientationNormalBank):
    def __init__(self,items,encoder_identity,binding,training_ids,training_hashes,available_views):
        if not items or len({x['group'] for x in items})<2:raise ValueError('Need at least two normal source groups')
        self.items,self.encoder_identity=items,encoder_identity
        self.binding,self.training_ids,self.training_hashes=binding,training_ids,training_hashes
        self.available_views=set(available_views);self.cls=[None]*8
        for view in available_views:
            self.cls[view]=F.normalize(torch.stack([x['views'][view]['cls'].float() for x in items]),dim=-1)

    def residual(self,feature,query_group=None,**kwargs):
        if not isinstance(query_group,tuple) or len(query_group)!=2:raise ValueError('Expected family and D4 orientation')
        if query_group[1] not in self.available_views:
            raise RuntimeError('Training D4 views must be regenerated from registered source weights/images; inference payload retains view0 only')
        return super().residual(feature,query_group=query_group,**kwargs)

    def write_recipe(self,folder):
        recipe={'format':'canonical_orientation_memory_v2','encoder':self.encoder_identity,
            'payload':self.binding,'selected_training_ids':self.training_ids,
            'rows':[{k:item[k] for k in ['id','group','cache_key','image_sha256','mask_sha256']} for item in self.items],
            'training_view_tensor_sha256':self.training_hashes,'inference_payload_views':[0],
            'orientation':'Eight genuinely re-encoded D4 views used during training; entire query capture family excluded. Exact view0 tensors persist for inference.',
            'reproduction':'Rebuild views1..7 from the registered merged encoder, selected source images/masks and original D4 transform; all8 tensor hashes must match before a resumed head fit.',
            'normal_eligibility':'Same canonical normal-only patch eligibility and256-frame farthest-first selection as parent study'}
        save_json(folder/'normal_bank_recipe.json',recipe);return recipe


def load_reference_bank(recipe_path,training_ids):
    recipe=read_json(recipe_path)
    if recipe.get('format')!='canonical_orientation_memory_v2':
        return OrientationNormalBank.from_recipe(recipe_path,training_ids)
    if set(recipe['selected_training_ids'])!=set(training_ids):raise PermissionError('Canonical memory budget differs from fitted source budget')
    if not {r['id'] for r in recipe['rows']}<=set(training_ids):raise PermissionError('Canonical memory contains a non-training reference')
    binding=recipe['payload'];path=(ROOT/binding['path']).resolve()
    if not path.is_relative_to(ROOT) or sha256(path)!=binding['sha256']:raise RuntimeError('Canonical reference payload changed')
    with gzip.open(path,'rb') as stream:
        data=torch.load(io.BytesIO(stream.read()),map_location='cpu',weights_only=True)
    if digest(data['encoder_identity'])!=digest(recipe['encoder']) or data['training_ids']!=recipe['selected_training_ids']:
        raise RuntimeError('Canonical reference identity changed')
    items=data['items']
    actual=[{k:item[k] for k in ['id','group','cache_key','image_sha256','mask_sha256']} for item in items]
    if actual!=recipe['rows'] or any(len(item['views'])!=1 for item in items):raise RuntimeError('Canonical reference rows/orientations changed')
    hashes={entry['id']:entry['view_sha256'] for entry in recipe['training_view_tensor_sha256']}
    if set(hashes)!={x['id'] for x in items} or any(len(values)!=8 for values in hashes.values()):
        raise RuntimeError('Training-view provenance is incomplete')
    if any(view_digest(x['views'][0])!=hashes[x['id']][0] for x in items):raise RuntimeError('Canonical view tensor differs from training')
    return PackedNormalBank(items,recipe['encoder'],binding,training_ids,recipe['training_view_tensor_sha256'],[0])


def build_packed_postadapt_bank(rows,encoder,size,identity,device,folder):
    """Rebuild all training orientations; keep view0 and hashes after exact verification."""
    folder=Path(folder).resolve()
    expected_root=ROOT/'artifacts/publication_20261017/mechanistic_controls_v1/adapted_encoders'
    if (not folder.is_relative_to(expected_root) or not identity.get('revision','').startswith('local:')
            or (ROOT/identity.get('model_id','')).resolve()!=folder
            or sha256(folder/'model.safetensors')!=identity['weights_sha256']):
        raise PermissionError('Only private post-adaptation export caches can be pruned')
    ids=[r['id'] for r in rows];metadata=folder/'packed_memory.json';existed=metadata.exists()
    bank=OrientationNormalBank.build(rows,encoder,size,identity,device)
    hashes=[{'id':item['id'],'view_sha256':[view_digest(v) for v in item['views']]} for item in bank.items]
    if metadata.exists():
        saved=read_json(metadata)
        if digest(saved['encoder'])!=digest(identity) or saved['selected_training_ids']!=ids or saved['training_view_tensor_sha256']!=hashes:
            raise RuntimeError('Regenerated source D4 normal memory differs from its frozen provenance')
        load_reference_bank(metadata,ids)  # Checks immutable canonical payload too.
        binding=saved['payload']
    else:
        path=folder/'normal_memory_view0.pt.gz';temporary=folder/'normal_memory_view0.tmp.pt.gz'
        canonical=[{**{k:v for k,v in item.items() if k!='views'},'views':[item['views'][0]]} for item in bank.items]
        with gzip.open(temporary,'wb',compresslevel=3) as stream:
            torch.save({'items':canonical,'encoder_identity':identity,'training_ids':ids},stream)
        temporary.replace(path);binding={'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}
    live=PackedNormalBank(bank.items,identity,binding,ids,hashes,range(8))
    recipe=live.write_recipe(folder)
    loaded=load_reference_bank(folder/'normal_bank_recipe.json',ids)
    for original,reloaded in zip(bank.items,loaded.items):
        for key in ['cls','patch','coords']:torch.testing.assert_close(original['views'][0][key],reloaded['views'][0][key],atol=0,rtol=0)
    removed=[]
    from . import publication_normal_bank as original_module
    for row in rows:
        key=digest({'encoder':identity,'input_size':size,'image':row['image_sha256'],
            'mask':row['mask_sha256'],'id':row['id'],'bank_code':sha256(original_module.__file__),
            'transform_code':sha256(ROOT/'elresearch/publication_dense_data.py')})
        cached=ROOT/'cache/publication_normal_d4'/(key+'.pt')
        if cached.exists():
            info=read_json(cached.with_suffix('.json'))
            if sha256(cached)!=info['payload_sha256']:raise RuntimeError('Adapted training cache changed before pruning')
            removed.append({'cache_key':key,'bytes':cached.stat().st_size,'sha256':info['payload_sha256']})
            cached.unlink();cached.with_suffix('.json').unlink()
    recipe['storage']={'canonical_payload_bytes':(ROOT/binding['path']).stat().st_size,
        'recoverable_intermediate_bytes_removed':sum(r['bytes'] for r in removed),
        'removed_owned_cache_rows':removed,'canonical_exact_reload_verified':True,
        'limits':'Only private derived adapted caches were pruned. All raw images, masks, official/merged weights, checkpoints, exact canonical references and8-view provenance remain.'}
    if existed:
        history=folder/'cache_regeneration_history.json'
        entries=read_json(history) if history.exists() else []
        entries.append({'at':now(),'all_eight_view_hashes_verified':True,'removed_owned_cache_rows':removed})
        save_json(history,entries)
    else:save_json(metadata,recipe)
    return live
