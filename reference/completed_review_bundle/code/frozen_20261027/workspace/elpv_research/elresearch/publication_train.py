"""Source-only matched-update training, calibration and immutable best checkpoints."""
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from .common import ROOT,digest,now,read_json,save_json,sha256,choose_device
from .data import load_image
from .encoders import image_tensor
from .image_classification import atomic_torch_save,trainable_state,restore_trainable,rng_state,restore_rng,assert_source_partitions
from .metrics import classification,recall_threshold
from .publication_models import build_backbone,PublicationClassifier,d4_view


FILES=['publication_models.py','publication_train.py','classification_heads.py','image_classification.py',
       'data.py','encoders.py','adapt.py','common.py','metrics.py']


def scientific_hashes():return {name:sha256(ROOT/'elresearch'/name) for name in FILES}


def budget_rows(rows,fraction,seed):
    """Nested class-stratified image annotations, with a fixed source validation pool."""
    if not 0<fraction<=1:raise ValueError('Invalid label fraction')
    rng=np.random.default_rng(seed);selected=[]
    for label in [0,1]:
        pool=sorted([r for r in rows if r['target']==label],key=lambda r:r['id'])
        if not pool:raise ValueError('Training pool lacks a class')
        count=max(1,int(round(fraction*len(pool))))
        selected.extend(pool[i] for i in rng.permutation(len(pool))[:count])
    return sorted(selected,key=lambda r:r['id'])


def round_plan(n,seed,round_index,steps,batch):
    rng=np.random.default_rng(np.random.SeedSequence([seed,round_index]))
    count=steps*batch
    order=np.concatenate([rng.permutation(n) for _ in range(math.ceil(count/n))])[:count]
    views=rng.integers(0,8,size=count)
    return order,views


def prepare_images(rows,size):
    return torch.stack([image_tensor(load_image(r),size)[0] for r in rows])


def feature_cache(model,identity,rows,settings,device):
    key=digest({'encoder':identity,'preprocessing_sha256':settings['scientific_hashes']['encoders.py'],
                'model_code_sha256':settings['scientific_hashes']['publication_models.py'],
                'images':[(r['id'],r['image_sha256']) for r in rows],
                'views':'rot90 k=0..3, optionally horizontal flip after rotation; float32 CLS'})
    path=ROOT/'cache/publication_cls_d4'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'],[r['id'] for r in rows])
            return torch.from_numpy(f['features'].copy())
    result=np.empty((len(rows),8,model.dimensions),np.float32);batch=settings['batch_size']
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),batch):
            images=prepare_images(rows[start:start+batch],settings['input_size']).to(device)
            for view in range(8):result[start:start+len(images),view]=model.features(d4_view(images,view)).cpu().numpy()
    if not np.isfinite(result).all():raise ValueError('Nonfinite frozen embeddings')
    temporary=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temporary,features=result,ids=np.array([r['id'] for r in rows]));temporary.replace(path)
    return torch.from_numpy(result)


def calibration(y,logits):
    y=np.asarray(y,np.float64);logits=np.asarray(logits,np.float64)
    def nll(log_t):
        z=logits/np.exp(log_t)
        return float(np.mean(np.logaddexp(0,z)-y*z))
    optimum=minimize_scalar(nll,bounds=(math.log(.05),math.log(20)),method='bounded')
    return float(np.exp(optimum.x))


def probability_metrics(y,logits,temperature=1.):
    y=np.asarray(y);prob=expit(np.asarray(logits,np.float64)/temperature)
    edges=np.linspace(0,1,11);ece=0.
    for i in range(10):
        mask=(prob>=edges[i])&((prob<edges[i+1]) if i<9 else (prob<=edges[i+1]))
        if mask.any():ece+=mask.mean()*abs(prob[mask].mean()-y[mask].mean())
    z=np.asarray(logits,np.float64)/temperature
    return {'brier':float(np.mean((prob-y)**2)),
            'negative_log_likelihood':float(np.mean(np.logaddexp(0,z)-y*z)),
            'ece_10_equal_width_bins':float(ece),'temperature':float(temperature)}


def predict(model,x,cached,device,batch_size):
    model.eval();result=[]
    with torch.inference_mode():
        for i in range(0,len(x),batch_size):
            values=x[i:i+batch_size].to(device)
            logits=model.classify(values[:,0]) if cached else model(values)
            result.append(logits.cpu().numpy())
    result=np.concatenate(result).astype(np.float64)
    if not np.isfinite(result).all():raise ValueError('Nonfinite logits')
    return result


def train(settings,spec,source_train,source_validation,seed,budget,folder):
    folder=Path(folder)
    if (folder/'run.json').exists():
        result=read_json(folder/'run.json')
        if result['checkpoint_sha256']!=sha256(folder/'best.pt'):raise RuntimeError('Completed checkpoint changed')
        return result
    if scientific_hashes()!=settings['scientific_hashes']:raise RuntimeError('Registered scientific implementation changed')
    assert_source_partitions(source_train,source_validation)
    selected=budget_rows(source_train,budget,seed)
    torch.set_num_threads(4);random.seed(seed);torch.manual_seed(seed)
    encoder_device=choose_device(settings['device'])
    backbone,dim,identity=build_backbone(spec['encoder'],settings['input_size'],encoder_device)
    model=PublicationClassifier(backbone,dim,spec['encoder'],spec['adaptation'],seed).to(encoder_device)
    cached=spec['adaptation']=='frozen'
    if cached:
        # Cache the complete permitted source pool once; nested budgets reuse identical views.
        all_rows=source_train+source_validation
        features=feature_cache(model,identity,all_rows,settings,encoder_device)
        index={row['id']:i for i,row in enumerate(source_train)}
        train_x=features[[index[row['id']] for row in selected]]
        val_x=features[len(source_train):]
        device=torch.device('cpu');model.to(device)
        if encoder_device.type=='mps':torch.mps.empty_cache()
    else:
        train_x,val_x=prepare_images(selected,settings['input_size']),prepare_images(source_validation,settings['input_size'])
        device=encoder_device
    y=np.array([r['target'] for r in source_validation]);target=torch.tensor([r['target'] for r in selected],dtype=torch.float32)
    signature={'study_id':settings['study_id'],'model':spec,'seed':seed,'training_budget_fraction':budget,
               'training_ids':[r['id'] for r in selected],'validation_ids':[r['id'] for r in source_validation],
               'encoder':identity,'scientific_hashes':settings['scientific_hashes'],
               'optimization_device':str(device),'encoder_device':str(encoder_device)}
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'started.json').exists():
        if digest(read_json(folder/'started.json')['signature'])!=digest(signature):raise RuntimeError('Interrupted run identity differs')
    else:save_json(folder/'started.json',{'signature':signature,'started_at':now()})
    optimizer=torch.optim.AdamW([
        {'params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':spec['encoder_lr']},
        {'params':model.head.parameters(),'lr':spec['head_lr']}],weight_decay=settings['weight_decay'])
    history=[];best=-float('inf');wait=0;elapsed=0.;first_grad=None
    if (folder/'latest.pt').exists():
        saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        restore_trainable(model,saved['parameters']);optimizer.load_state_dict(saved['optimizer'])
        history,best,wait,elapsed=saved['history'],saved['best'],saved['wait'],saved['elapsed']
        first_grad=saved['first_grad'];restore_rng(saved['rng'],device)
    batch=settings['batch_size'];steps=settings['updates_per_round'];total_steps=settings['maximum_rounds']*steps
    started=time.perf_counter()
    for round_index in range(len(history),settings['maximum_rounds']):
        if wait>=settings['patience']:break
        model.train();order,views=round_plan(len(selected),seed,round_index,steps,batch)
        losses=[];round_start=time.perf_counter()
        for update in range(steps):
            k=slice(update*batch,(update+1)*batch);idx,view=order[k],views[k]
            factor=.1+.9*(1+math.cos(math.pi*(round_index*steps+update)/total_steps))/2
            for group,base in zip(optimizer.param_groups,[spec['encoder_lr'],spec['head_lr']]):group['lr']=base*factor
            if cached:
                values=train_x[idx,view].to(device);logits=model.classify(values)
            else:
                values=torch.stack([d4_view(train_x[int(i)],int(v)) for i,v in zip(idx,view)]).to(device)
                logits=model(values)
            loss=F.binary_cross_entropy_with_logits(logits,target[idx].to(device))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if first_grad is None:
                first_grad={part:{'parameters_with_gradient':sum(p.numel() for p in module.parameters() if p.grad is not None),
                                 'gradient_l2':float(sum(p.grad.detach().square().sum().cpu() for p in module.parameters() if p.grad is not None)**.5)}
                            for part,module in [('encoder',model.encoder),('head',model.head)]}
                if first_grad['head']['gradient_l2']<=0 or (not cached and first_grad['encoder']['gradient_l2']<=0):
                    raise RuntimeError('Gradient contract violated')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
        logits=predict(model,val_x,cached,device,batch);auc=float(roc_auc_score(y,logits))
        history.append({'round':round_index+1,'optimizer_updates':steps*(round_index+1),
                        'training_bce':float(np.mean(losses)),'validation_auroc':auc,
                        'validation_bce':probability_metrics(y,logits)['negative_log_likelihood'],
                        'seconds':time.perf_counter()-round_start})
        if auc>best+1e-12:
            best,wait=auc,0;atomic_torch_save(trainable_state(model),folder/'best.pt')
        else:wait+=1
        save_json(folder/'history.json',history)
        atomic_torch_save({'parameters':trainable_state(model),'optimizer':optimizer.state_dict(),
                          'history':history,'best':best,'wait':wait,'elapsed':elapsed+time.perf_counter()-started,
                          'first_grad':first_grad,'rng':rng_state(device)},folder/'latest.pt')
        print(f"{spec['name']} seed={seed} budget={budget:g} round={round_index+1}: "
              f"BCE={history[-1]['training_bce']:.4f} source-val AUROC={auc:.4f} "
              f"seconds={history[-1]['seconds']:.1f}",flush=True)
    restore_trainable(model,torch.load(folder/'best.pt',map_location='cpu',weights_only=True))
    logits=predict(model,val_x,cached,device,batch)
    threshold=recall_threshold(y,logits,settings['target_source_recall'])
    temperature=calibration(y,logits)
    result={**signature,'completed_at':now(),'training_images':len(selected),'validation_images':len(source_validation),
            'training_capture_groups':len({r['split_group'] for r in selected}),
            'total_labeled_images_including_validation':len(selected)+len(source_validation),
            'source_training_pool_images':len(source_train),'cached_augmented_features':cached,
            'trainable_encoder_parameters':sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
            'head_parameters':sum(p.numel() for p in model.head.parameters()),
            'total_parameters':sum(p.numel() for p in model.parameters()),'first_gradient_check':first_grad,
            'validation_metrics':classification(y,logits,threshold),
            'validation_metrics_at_zero_logit':classification(y,logits,0.),
            'validation_calibration_raw':probability_metrics(y,logits),
            'validation_calibration_temperature_scaled':probability_metrics(y,logits,temperature),
            'source_calibrated_logit_threshold':threshold,'source_temperature':temperature,
            'best_round':max(history,key=lambda r:r['validation_auroc'])['round'],
            'completed_rounds':len(history),'optimizer_updates':len(history)*steps,
            'optimization_seconds':elapsed+time.perf_counter()-started,
            'checkpoint_sha256':sha256(folder/'best.pt'),
            'supervision':'image crack-presence labels only; validation labels charged separately',
            'selection':'source validation AUROC on raw logits; earliest round on ties',
            'scope':settings['scope']}
    np.savez_compressed(folder/'validation_predictions.npz',ids=np.array([r['id'] for r in source_validation]),
                        y=y,logits=logits,groups=np.array([r['split_group'] for r in source_validation]))
    save_json(folder/'run.json',result);(folder/'latest.pt').unlink(missing_ok=True)
    print('COMPLETED',spec['name'],seed,'AUROC',result['validation_metrics']['auc'],flush=True)
    del model,backbone,train_x,val_x
    if encoder_device.type=='mps':torch.mps.empty_cache()
    return result
