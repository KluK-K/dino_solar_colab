"""Full published-architecture source training, including teacher EMA and BN buffers."""
import gc
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score

from .common import ROOT,read_json,save_json,sha256,digest,now,choose_device
from .data import IGNORE
from .image_classification import assert_source_partitions,atomic_torch_save,rng_state,restore_rng
from .publication_train import budget_rows,round_plan,calibration,probability_metrics
from .publication_dense_data import prepare_dense_rows,augmented_batch
from .publication_dense_train import selection_predictions,native_evaluation,FILES as DENSE_FILES
from .metrics import classification,recall_threshold
from .published_semisup import PublishedTrainSystem,published_identity,inference_state,ema_update,confidence_ce
from .published_data import prepare_unlabelled,weak_unlabelled_batch,strong_photometric,cutmix_unlabelled
from .published_archive import inference_basis,install_basis_heads,load_published_inference
from .lossless_state import save_lossless_state,load_lossless_state


FILES=list(dict.fromkeys(DENSE_FILES+['published_weights.py','published_psmt_resnet.py',
    'published_psmt_architecture.py','published_semisup.py','published_data.py',
    'published_archive.py','lossless_state.py','published_train.py','published_evaluate.py','publication_splits.py']))


def scientific_hashes():return {name:sha256(ROOT/'elresearch'/name) for name in FILES}


class TrainingView(torch.nn.Module):
    def __init__(self,system,view='primary'):
        super().__init__();self.first=system.first if system.semi else system.student
        if system.semi:self.second=system.second
        self.semi,self.view=system.semi,view
    def forward(self,images,valid,geometries=None,query_groups=None):
        from .full_models import top_evidence
        logits=self.first(images)
        if self.semi and self.view=='ensemble':logits=.5*(logits+self.second(images))
        return {'image':top_evidence(logits,valid,.05),'segmentation':logits}


def save_selection(folder,name,logits,scores):
    path=folder/name;temporary=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temporary,image_logits=logits,pixel_scores=scores);temporary.replace(path)


def train_published(settings,spec,source_train,source_validation,seed,budget,folder):
    folder=Path(folder)
    if scientific_hashes()!=settings['scientific_hashes']:raise RuntimeError('Published-method scientific implementation changed')
    assert_source_partitions(source_train,source_validation)
    if any(r['partition']!='train' for r in source_train) or any(r['partition']!='validation' for r in source_validation):
        raise PermissionError('Published-method fitting is source-only')
    if (folder/'run.json').exists():
        result=read_json(folder/'run.json')
        if result['checkpoint_sha256']!=sha256(folder/'best.pt') or result['study_id']!=settings['study_id']:
            raise RuntimeError('Completed published-method source model changed')
        return result
    selected=budget_rows(source_train,budget,seed);selected_ids={r['id'] for r in selected}
    unlabelled=[r for r in source_train if r['id'] not in selected_ids]
    semi=spec['semi']
    if semi and not unlabelled:raise ValueError('A semi-supervised fit needs a nonempty image-only source pool')
    random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    device=choose_device(settings['device']);size=settings['input_size'];batch=settings['batch_size']
    basis_model,basis,binding=inference_basis(seed,size,semi)
    inference_parameters=sum(p.numel() for p in basis_model.first.parameters())
    loaded_parameters=sum(p.numel() for p in basis_model.parameters());del basis_model;gc.collect()
    system=PublishedTrainSystem(seed,size,semi).to(device)
    install_basis_heads(system.student,binding)
    training=prepare_dense_rows(selected,size);validation=prepare_dense_rows(source_validation,size)
    image_pool=prepare_unlabelled(unlabelled,size) if semi else []
    steps=max(settings['minimum_updates_per_round'],math.ceil(len(source_train)/batch))
    if settings.get('implementation_smoke'):steps=settings.get('smoke_updates_per_round',1)
    identity=published_identity(size,semi)
    model_spec={**spec,'architecture':'published_psmt_deeplab','archive_basis':binding,
        'backbone_sha256':identity['backbone']['sha256'],'checkpoint_format':'xor_byteshuffle_gzip_v1'}
    signature={'study_id':settings['study_id'],'model':model_spec,'seed':seed,'input_size':size,
        'training_budget_fraction':budget,'training_ids':[r['id'] for r in selected],
        'validation_ids':[r['id'] for r in source_validation],
        'unlabelled_source_ids':[r['id'] for r in unlabelled] if semi else [],
        'encoder':identity,'scientific_hashes':settings['scientific_hashes'],'device':str(device),
        'updates_per_round':steps,'maximum_rounds':settings['maximum_rounds'],
        'implementation_smoke':bool(settings.get('implementation_smoke',False)),
        'initialization':'Student uses author-posted ImageNet backbone; teachers initialize randomly as in released PV-S3 roles. Independent seed streams are explicit.'}
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'started.json').exists():
        if digest(read_json(folder/'started.json')['signature'])!=digest(signature):raise RuntimeError('Interrupted published-method fit identity changed')
    else:save_json(folder/'started.json',{'at':now(),'signature':signature})
    optimizer=torch.optim.SGD(system.student.parameters(),lr=spec['learning_rate'],momentum=.9,weight_decay=1e-4)
    history=[];best=-np.inf;wait=0;elapsed=0.;gradient_check=None;storage=None
    if (folder/'latest.pt').exists():
        saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        if saved['signature_digest']!=digest(signature):raise RuntimeError('Resume signature changed')
        system.load_state_dict(saved['state'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        history,best,wait,elapsed,gradient_check,storage=[saved[k] for k in ['history','best','wait','elapsed','gradient_check','storage']]
        if sha256(folder/'best.pt')!=saved['best_checkpoint_sha256']:raise RuntimeError('Selected checkpoint changed on resume')
        restore_rng(saved['rng'],device)
    y=np.array([r['target'] for r in source_validation]);total=steps*settings['maximum_rounds'];started=time.perf_counter()
    for round_index in range(len(history),settings['maximum_rounds']):
        if wait>=settings['patience']:break
        system.train();active=1 if round_index%2==0 else 2
        order,views=round_plan(len(training),seed,round_index,steps,batch)
        if semi:ul_order,ul_views=round_plan(len(image_pool),seed+50000,round_index,steps,batch)
        rng=np.random.default_rng(np.random.SeedSequence([seed,round_index,72319]))
        losses=[];supervised_losses=[];unsupervised_losses=[];coverage=[];vat_norms=[];clock=time.perf_counter()
        for update in range(steps):
            index=slice(update*batch,(update+1)*batch)
            b=augmented_batch([training[int(i)] for i in order[index]],views[index],device)
            images=strong_photometric(b['images'],rng)
            learning_rate=spec['learning_rate']*(1-(round_index*steps+update)/total)**.9
            for group in optimizer.param_groups:group['lr']=learning_rate
            logits,vat=system.supervised(images,use_vat=semi,xi=spec.get('vat_xi',.1))
            supervised=F.cross_entropy(logits,b['target'],ignore_index=IGNORE);unsupervised=supervised*0
            if vat:vat_norms.append(vat['adversarial_norm_mean'])
            confident=0.;ramp=0.
            if semi:
                u=weak_unlabelled_batch([image_pool[int(i)] for i in ul_order[index]],ul_views[index],device)
                pseudo=system.pseudo(u['images'],active)
                strong=strong_photometric(u['images'],rng)
                strong,pseudo,valid=cutmix_unlabelled(strong,pseudo,u['valid'],rng)
                out,ul_vat=system.supervised(strong,use_vat=True,xi=spec.get('vat_xi',.1))
                unsupervised,info=confidence_ce(out,pseudo,valid,.6,spec.get('negative_threshold',0.))
                progress=min(1.,(round_index*steps+update)/(12*steps));ramp=1.5*math.exp(-5*(1-progress)**2)
                confident=info['confident_pixel_fraction'];vat_norms.append(ul_vat['adversarial_norm_mean'])
            loss=supervised+ramp*unsupervised
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite supervised/teacher consistency loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if gradient_check is None:
                gradient_check={'student_backbone_gradient':any(p.grad is not None and p.grad.abs().sum()>0 for p in system.student.encoder.backbone.parameters()),
                    'student_decoder_gradient':any(p.grad is not None and p.grad.abs().sum()>0 for p in system.student.decoder.parameters()),
                    'teacher_gradient':any(p.grad is not None for model in ([system.first,system.second] if semi else []) for p in model.parameters()),
                    'student_BN_buffers_saved':True}
                if not gradient_check['student_backbone_gradient'] or not gradient_check['student_decoder_gradient'] or gradient_check['teacher_gradient']:
                    raise RuntimeError('Published-method gradient/frozen-teacher contract violated')
            optimizer.step()
            if semi:ema_update(system.first if active==1 else system.second,system.student,.996)
            losses.append(float(loss.detach().cpu()));supervised_losses.append(float(supervised.detach().cpu()))
            unsupervised_losses.append(float(unsupervised.detach().cpu()));coverage.append(confident)
        primary=TrainingView(system).eval();logits,scores,truth=selection_predictions(primary,validation,device,batch)
        ap=float(average_precision_score(truth,scores))
        history.append({'round':round_index+1,'optimizer_updates':(round_index+1)*steps,'training_loss':float(np.mean(losses)),
            'supervised_CE':float(np.mean(supervised_losses)),'unsupervised_ConfCE':float(np.mean(unsupervised_losses)),
            'pseudo_confident_fraction':float(np.mean(coverage)),'VAT_norm_mean':float(np.mean(vat_norms)) if vat_norms else None,
            'primary_source_selection_pixel_AP':ap,'primary_source_image_AUROC':float(roc_auc_score(y,logits)),
            'active_teacher':active if semi else None,'seconds':time.perf_counter()-clock})
        if ap>best+1e-12:
            best,wait=ap,0
            storage=save_lossless_state(folder/'best.pt',inference_state(system),basis,binding)
            save_selection(folder,'selected_source_predictions.npz',logits,scores)
            if semi:
                aux,aux_scores,_=selection_predictions(TrainingView(system,'ensemble').eval(),validation,device,batch)
                save_selection(folder,'selected_source_predictions_ensemble.npz',aux,aux_scores)
        else:wait+=1
        save_json(folder/'history.json',history)
        atomic_torch_save({'signature_digest':digest(signature),'state':system.state_dict(),'optimizer':optimizer.state_dict(),
            'history':history,'best':best,'wait':wait,'gradient_check':gradient_check,'storage':storage,
            'best_checkpoint_sha256':sha256(folder/'best.pt'),'elapsed':elapsed+time.perf_counter()-started,'rng':rng_state(device)},folder/'latest.pt')
        print('PUBLISHED',spec['name'],seed,budget,'round',round_index+1,'pixel AP',round(ap,4),'seconds',round(history[-1]['seconds'],1),flush=True)
    elapsed+=time.perf_counter()-started
    trainable=sum(p.numel() for p in system.student.parameters());total_parameters=sum(p.numel() for p in system.parameters())
    del system,optimizer,primary,training,image_pool;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    run={**signature,'completed_at':now(),'training_images':len(selected),'validation_images':len(source_validation),
        'total_pixel_annotated_images_including_validation':len(selected)+len(source_validation),
        'unlabelled_source_images':len(unlabelled) if semi else 0,'trainable_parameters':trainable,'total_training_system_parameters':total_parameters,
        'first_gradient_check':gradient_check,'completed_rounds':len(history),'optimizer_updates':len(history)*steps,
        'best_round':max(history,key=lambda h:h['primary_source_selection_pixel_AP'])['round'],
        'checkpoint_sha256':sha256(folder/'best.pt'),'lossless_checkpoint':storage,
        'training_and_validation_seconds_excluding_preparation':elapsed,
        'supervision':'Selected canonical source pixel masks; no image BCE. Only the remaining source-training IMAGES supply SSL consistency, never their masks/labels.',
        'selection':'Primary teacher1 (SSL) or supervised student; fixed sampled source-validation pixel AP; earliest tie. Ensemble is an additional fixed view at the same checkpoint.',
        'scope':'Documented complete published architecture/mechanisms port; not the original experiment or a reproduction of its reported score.'}
    summaries={}
    for view in (['primary','ensemble'] if semi else ['primary']):
        model=load_published_inference(run,folder/'best.pt',device,view)
        destination=folder if view=='primary' else folder/'inference_views'/view;destination.mkdir(parents=True,exist_ok=True)
        logits,local=native_evaluation(model,validation,device,batch,destination)
        again,again_scores,_=selection_predictions(model,validation,device,batch)
        original_path=folder/('selected_source_predictions.npz' if view=='primary' else 'selected_source_predictions_ensemble.npz')
        with np.load(original_path,allow_pickle=False) as original:
            np.testing.assert_allclose(original['image_logits'],again,atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(original['pixel_scores'],again_scores,atol=2e-5,rtol=2e-5)
            proof={'fresh_model':True,'logit_max_abs_error':float(np.max(abs(original['image_logits']-again))),
                'sampled_pixel_probability_max_abs_error':float(np.max(abs(original['pixel_scores']-again_scores)))}
        threshold=recall_threshold(y,logits,settings['target_source_recall']);temperature=calibration(y,logits)
        summaries[view]={'source_logit_threshold':threshold,'source_temperature':temperature,
            'validation_metrics':classification(y,logits,threshold),'validation_localization':local,
            'validation_calibration_temperature_scaled':probability_metrics(y,logits,temperature),'checkpoint_reload':proof}
        prediction_name='validation_predictions.npz' if view=='primary' else 'validation_predictions_ensemble.npz'
        np.savez_compressed(folder/prediction_name,ids=np.array([r['id'] for r in source_validation]),
            y=y,logits=logits,groups=np.array([r['split_group'] for r in source_validation]))
        del model;gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
    run.update(summaries['primary']);run['inference_views']=summaries
    run['inference_parameters']=inference_parameters
    run['loaded_inference_parameters']=loaded_parameters
    run['inference_parameters_by_view']={'primary':inference_parameters,**({'ensemble':loaded_parameters} if semi else {})}
    save_json(folder/'run.json',run);(folder/'latest.pt').unlink(missing_ok=True)
    return run
