"""Full-network training; source-only early stopping and native-mask validation."""
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from .common import ROOT, config, choose_device, digest, save_json, read_json, sha256, now
from .data import records,load_image,load_mask,IGNORE
from .encoders import FrozenEncoder,image_tensor,reproject,extract
from .heads import NormalBank,budget_rows
from .full_models import FullDINO,FullDeepLab,dense_loss
from .metrics import classification,recall_threshold
from .classification_heads import head_spec

def dense_target(mask,g):
    value=torch.from_numpy(mask.astype(np.float32))[None,None]
    resized=F.interpolate(value,size=(g['h'],g['w']),mode='nearest')[0,0].long()
    # Preserve annotated cracks when downsampling; all models receive the same targets.
    thin=F.adaptive_max_pool2d(torch.from_numpy((mask==1).astype(np.float32))[None,None],(g['h'],g['w']))[0,0]>0
    resized[thin]=1
    canvas=torch.full((g['size'],g['size']),IGNORE,dtype=torch.long)
    canvas[g['top']:g['top']+g['h'],g['left']:g['left']+g['w']]=resized
    return canvas

def prepare_dense(rows,size):
    result=[]
    for row in rows:
        image,g=image_tensor(load_image(row),size)
        mask=load_mask(row)
        target=dense_target(mask,g)
        # Geometric validity is independent of which annotation regions are ignored.
        valid=torch.zeros(size,size,dtype=torch.bool)
        valid[g['top']:g['top']+g['h'],g['left']:g['left']+g['w']]=True
        result.append({'row':row,'image':image,'geometry':g,'target':target,'valid':valid,'native_mask':mask})
    return result

def dense_batch(items,device):
    return {'images':torch.stack([x['image'] for x in items]).to(device),
            'valid':torch.stack([x['valid'] for x in items]).to(device),
            'target':torch.stack([x['target'] for x in items]).to(device),
            'y':torch.tensor([x['row']['target'] for x in items],dtype=torch.float32,device=device),
            'geometries':[x['geometry'] for x in items], 'groups':[x['row']['split_group'] for x in items]}

def dense_predict(model,examples,device,batch_size,with_maps=True):
    model.eval();scores=[];maps=[]
    with torch.inference_mode():
        for start in range(0,len(examples),batch_size):
            selected=examples[start:start+batch_size];b=dense_batch(selected,device)
            out=model(b['images'],b['valid'],b['geometries'],b['groups'])
            scores.extend(out['image'].sigmoid().cpu().numpy())
            if with_maps:
                maps.extend(out['segmentation'].softmax(1)[:,1].cpu().numpy())
    return np.asarray(scores),maps

def native_localization(examples,maps,pixel_threshold=.5):
    truth=[];probability=[];dice=[];iou=[];positive_dice=[]
    for item,values in zip(examples,maps):
        prediction=reproject(values,item['geometry']);mask=item['native_mask']
        valid=mask!=IGNORE;true=(mask==1)&valid;hard=(prediction>pixel_threshold)&valid
        truth.append(true[valid]);probability.append(prediction[valid])
        intersection=int((true&hard).sum());total=int(true.sum()+hard.sum());union=int((true|hard).sum())
        d=2*intersection/total if total else 1.;u=intersection/union if union else 1.
        dice.append(d);iou.append(u)
        if true.any():positive_dice.append(d)
    return {'pooled_pixel_AP':float(average_precision_score(np.concatenate(truth),np.concatenate(probability))),
            'mean_Dice_all_images':float(np.mean(dice)),'mean_IoU_all_images':float(np.mean(iou)),
            'mean_Dice_positive_images':float(np.mean(positive_dice)),'pixel_threshold':pixel_threshold,
            'pixel_threshold_origin':'fixed in protocol; not fitted on target','space':'native annotation pixels'}

def train_dense(dataset='ucf',architecture='dinov3_decoder',adaptation='frozen',reference=False,
                seed=11,budget=1.,epochs=40,batch_size=2,device='auto',size=336,patience=6,image_head='linear'):
    if dataset=='elpv':raise ValueError('ELPV does not have crack masks')
    cfg=config();torch.manual_seed(seed);rng=np.random.default_rng(seed);device=choose_device(device)
    train=budget_rows(records(dataset,'train'),budget,seed);validation=records(dataset,'validation')
    manifest=read_json(ROOT/'data'/dataset/'manifest.json')
    bank=None;encoder_identity=None;encoder_key=None
    if architecture=='dinov3_decoder':
        encoder=FrozenEncoder('dinov3_small',size,str(device));encoder_identity=encoder.identity;encoder_key=encoder.key
        if reference:
            if adaptation!='frozen':raise ValueError('Rebuild normal bank after adaptation; simultaneous stale-bank LoRA is prohibited')
            extract(dataset,encoder,'train',cfg['batch_size'])
            bank=NormalBank.build(train,encoder,cfg['reference_images'])
        model=FullDINO(encoder.model,bank,adaptation,reference_neighbors=cfg['reference_neighbors'],geometry_weight=cfg['geometry_weight'],top_fraction=cfg['top_fraction'],image_head=image_head).to(device)
    elif architecture=='deeplabv3_resnet50':
        if image_head!='linear':raise ValueError('DeepLab head is fixed; use the image-only study for matched head comparisons')
        if reference or adaptation!='full':raise ValueError('DeepLab baseline uses full network fine-tuning, without normal references')
        model=FullDeepLab(pretrained=True,top_fraction=cfg['top_fraction']).to(device)
        weight=Path(torch.hub.get_dir())/'checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth'
        encoder_identity={'name':'deeplabv3_resnet50','pretraining':'COCO_WITH_VOC_LABELS_V1','weights_sha256':sha256(weight)}
    else:raise ValueError(architecture)
    signature={'dataset':dataset,'architecture':architecture,'adaptation':adaptation,'reference':reference,
               'encoder':encoder_identity,'encoder_key':encoder_key,'source_manifest':manifest['signature'],
               'seed':seed,'budget':budget,'epochs':epochs,'batch_size':batch_size,'input_size':size,'patience':patience,
               'learning_rates':{'backbone_or_lora':1e-4,'new_heads':1e-3},'weight_decay':cfg['weight_decay'],
               'reference_config':{'neighbors':cfg['reference_neighbors'],'geometry_weight':cfg['geometry_weight'],
                                   'maximum_images':cfg['reference_images']},'top_fraction':cfg['top_fraction'],
               'image_head':head_spec(2*model.dimensions+1,image_head) if architecture=='dinov3_decoder' else head_spec(1),
               'loss':'image BCE + weighted pixel CE + 0.5 crack Dice; DeepLab auxiliary CE weight 0.4',
               'augmentation':'none in initial architecture pilot','selection':'source validation image AUROC',
               'code':{'models':sha256(ROOT/'elresearch/full_models.py'),'training':sha256(__file__),
                       'classification_heads':sha256(ROOT/'elresearch/classification_heads.py')}}
    run_id=digest(signature)[:20];folder=ROOT/'artifacts/dense_runs'/run_id
    if (folder/'run.json').exists() or (folder/'frozen.json').exists():raise RuntimeError('Completed full-model run is immutable')
    folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'started.json',{**signature,'run_id':run_id,'started_at':now(),'training_ids':[r['id'] for r in train]})
    if bank:bank.save(folder/'normal_bank.pt')
    train_items=prepare_dense(train,size);validation_items=prepare_dense(validation,size)
    counts=sum((torch.bincount(x['target'][x['target']!=IGNORE],minlength=3) for x in train_items),torch.zeros(3,dtype=torch.long))
    weights=counts.clamp_min(1).float().rsqrt();weights=weights/weights.mean();weights=weights.to(device)
    backbone=[];heads=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            (backbone if name.startswith('encoder.') or name.startswith('network.backbone.') else heads).append(p)
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':1e-4},{'params':heads,'lr':1e-3}],weight_decay=cfg['weight_decay'])
    started=time.perf_counter();best=-np.inf;wait=0;history=[]
    for epoch in range(epochs):
        model.train();losses=[];order=rng.permutation(len(train_items));epoch_start=time.perf_counter()
        for start in range(0,len(order),batch_size):
            items=[train_items[i] for i in order[start:start+batch_size]];b=dense_batch(items,device)
            output=model(b['images'],b['valid'],b['geometries'],b['groups'])
            loss=dense_loss(output,b['target'],b['y'],weights)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite dense loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scores,_=dense_predict(model,validation_items,device,batch_size,False)
        auc=classification([r['target'] for r in validation],scores,.5)['auc']
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_auc':auc,'seconds':time.perf_counter()-epoch_start})
        print(f'{architecture}/{adaptation}/reference={reference} epoch {epoch+1}: loss={np.mean(losses):.4f}, val AUC={auc:.4f}, seconds={history[-1]["seconds"]:.1f}',flush=True)
        save_json(folder/'history.json',history)
        if auc>best:
            best=auc;wait=0;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},folder/'model.pt')
        else:
            wait+=1
            if wait>=patience:break
    model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True))
    scores,maps=dense_predict(model,validation_items,device,batch_size)
    y=np.array([r['target'] for r in validation]);threshold=recall_threshold(y,scores,cfg['review_recall'])
    local=native_localization(validation_items,maps)
    for item,value in zip(validation_items,maps):
        dest=folder/'validation_maps'/(digest(item['row']['id'])+'.npz');dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,score=reproject(value,item['geometry']).astype(np.float16))
    result={**signature,'run_id':run_id,'kind':'dense','created_at':now(),'training_ids':[r['id'] for r in train],
            'calibration_ids':[r['id'] for r in validation],'threshold':threshold,'validation_metrics':classification(y,scores,threshold),
            'validation_localization':local,'checkpoint_sha256':sha256(folder/'model.pt'),
            'normal_bank_sha256':sha256(folder/'normal_bank.pt') if bank else None,
            'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'train_images':len(train),'validation_images':len(validation),'seconds':time.perf_counter()-started,
            'publication_eligible':manifest.get('publication_eligible',False),'scope':'source-development architecture pilot; not original paper reproduction',
            'device':str(device),'supervision':'same source image labels and 3-class canonical pixel masks'}
    save_json(folder/'run.json',result)
    np.savez_compressed(folder/'validation_predictions.npz',ids=np.array([r['id'] for r in validation]),y=y,scores=scores,
                        groups=np.array([r['split_group'] for r in validation]))
    print('DENSE RUN COMPLETE',run_id,local,flush=True)
    return result
