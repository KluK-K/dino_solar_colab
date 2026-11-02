"""Explicit mechanism controls using the registered dense optimizer and selection recipe.

Training loop copied from publication_dense_train.py so the queued parent study stays
immutable. This module has its own source hash and source-only run identities.
"""
import gc
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score

from .common import ROOT,digest,sha256,now,save_json,read_json,choose_device
from .data import IGNORE
from .encoders import patch_targets
from .full_models import dense_loss
from .image_classification import (atomic_torch_save,trainable_state,restore_trainable,rng_state,restore_rng,assert_source_partitions)
from .metrics import classification,recall_threshold
from .publication_train import budget_rows,round_plan,calibration,probability_metrics
from .publication_dense_data import prepare_dense_rows,augmented_batch
from .publication_models import d4_view
from .publication_dense_train import selection_predictions,native_evaluation,FILES as PARENT_FILES
from .mechanistic_models import build_mechanistic_system as build_system

FILES=list(dict.fromkeys(PARENT_FILES+['mechanistic_models.py','mechanistic_train.py','postadapt_encoder.py',
    'normal_distance_fit.py','mechanistic_evaluate.py','packed_normal_bank.py','publication_splits.py']))


def scientific_hashes():return {name:sha256(ROOT/'elresearch'/name) for name in FILES}


def prepare_mechanistic_rows(rows,size):
    items=prepare_dense_rows(rows,size)
    for item in items:
        item['patch_target']=torch.from_numpy(patch_targets(item['native_mask'],item['geometry'],16))
        if not (item['patch_target']!=IGNORE).any():raise ValueError('No known image-covered patch target')
    return items


def mechanistic_batch(items,codes,device):
    batch=augmented_batch(items,codes,device)
    batch['patch_target']=torch.stack([d4_view(item['patch_target'],int(code)) for item,code in zip(items,codes)]).to(device)
    return batch


def mechanistic_loss(output,batch,weights):
    if 'patch_logits' not in output:return dense_loss(output,batch['target'],batch['y'],weights)
    logits=output['patch_logits'];target=batch['patch_target'];valid=target!=IGNORE
    ce=F.cross_entropy(logits,target,weight=weights,ignore_index=IGNORE)
    probability=logits.softmax(1)[:,1]*valid;truth=(target==1).float()
    intersection=(probability*truth).sum((1,2))
    dice=1-((2*intersection+1)/(probability.sum((1,2))+truth.sum((1,2))+1)).mean()
    return F.binary_cross_entropy_with_logits(output['image'],batch['y'])+ce+.5*dice


def train(settings, spec, source_train, source_validation, seed, budget, folder):
    folder = Path(folder)
    if scientific_hashes() != settings['scientific_hashes']:
        raise RuntimeError('Registered dense scientific implementation changed')
    assert_source_partitions(source_train, source_validation)
    if any(r['partition'] != 'train' for r in source_train) or any(r['partition'] != 'validation' for r in source_validation):
        raise PermissionError('Dense training must use registered source training/validation rows')
    if (folder / 'run.json').exists():
        result = read_json(folder / 'run.json')
        if sha256(folder / 'best.pt') != result['checkpoint_sha256']:
            raise RuntimeError('Completed dense checkpoint changed')
        return result
    selected = budget_rows(source_train, budget, seed)
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    device = choose_device(settings['device'])
    model, identity, bank = build_system(spec, seed, settings['input_size'], device, selected)
    signature = {'study_id': settings['study_id'], 'model': spec, 'seed': seed,
                 'training_budget_fraction': budget, 'training_ids': [r['id'] for r in selected],
                 'validation_ids': [r['id'] for r in source_validation], 'encoder': identity,
                 'scientific_hashes': settings['scientific_hashes'], 'device': str(device)}
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'started.json').exists():
        if digest(read_json(folder / 'started.json')['signature']) != digest(signature):
            raise RuntimeError('Interrupted dense run identity changed')
    else:
        save_json(folder / 'started.json', {'signature': signature, 'started_at': now()})
    if bank:
        bank.write_recipe(folder)
    training = prepare_mechanistic_rows(selected, settings['input_size'])
    validation = prepare_mechanistic_rows(source_validation, settings['input_size'])
    weight_key = 'patch_target' if spec['architecture'] == 'dinov3_local_patch' else 'target'
    counts = sum((torch.bincount(x[weight_key][x[weight_key] != IGNORE], minlength=3) for x in training), torch.zeros(3, dtype=torch.long))
    weights = counts.clamp_min(1).float().rsqrt(); weights = (weights / weights.mean()).to(device)
    head = [p for name, p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
    backbone = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': spec['encoder_lr']},
                                   {'params': head, 'lr': spec['head_lr']}], weight_decay=settings['weight_decay'])
    history, best, wait, elapsed, first_grad = [], -np.inf, 0, 0., None
    if (folder / 'latest.pt').exists():
        saved = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        restore_trainable(model, saved['parameters']); optimizer.load_state_dict(saved['optimizer'])
        history, best, wait, elapsed, first_grad = [saved[k] for k in ['history', 'best', 'wait', 'elapsed', 'first_grad']]
        restore_rng(saved['rng'], device)
    batch, steps = settings['batch_size'], settings['updates_per_round']
    total_steps = settings['maximum_rounds'] * steps
    y = np.array([x['target'] for x in source_validation])
    started = time.perf_counter()
    for round_index in range(len(history), settings['maximum_rounds']):
        if wait >= settings['patience']:
            break
        model.train(); order, views = round_plan(len(training), seed, round_index, steps, batch)
        losses = []; round_started = time.perf_counter()
        for update in range(steps):
            k = slice(update * batch, (update + 1) * batch)
            b = mechanistic_batch([training[int(i)] for i in order[k]], views[k], device)
            factor = .1 + .9 * (1 + math.cos(math.pi * (round_index * steps + update) / total_steps)) / 2
            for group, base in zip(optimizer.param_groups, [spec['encoder_lr'], spec['head_lr']]):
                group['lr'] = base * factor
            output = model(b['images'], b['valid'], b['geometries'], b['groups'])
            loss = mechanistic_loss(output, b, weights)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite dense loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if first_grad is None:
                first_grad = {'encoder_has_gradient': any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone),
                              'heads_have_gradient': any(p.grad is not None and p.grad.abs().sum() > 0 for p in head)}
                if not first_grad['heads_have_gradient'] or (bool(backbone) != first_grad['encoder_has_gradient']):
                    raise RuntimeError('Dense gradient contract violated')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        logits, score, truth = selection_predictions(model, validation, device, batch)
        ap = float(average_precision_score(truth, score))
        history.append({'round': round_index + 1, 'optimizer_updates': (round_index + 1) * steps,
                        'training_joint_loss': float(np.mean(losses)), 'validation_selection_pixel_AP': ap,
                        'validation_image_AUROC': float(roc_auc_score(y, logits)),
                        'seconds': time.perf_counter() - round_started})
        if ap > best + 1e-12:
            best, wait = ap, 0; atomic_torch_save(trainable_state(model), folder / 'best.pt')
        else:
            wait += 1
        save_json(folder / 'history.json', history)
        atomic_torch_save({'parameters': trainable_state(model), 'optimizer': optimizer.state_dict(),
                          'history': history, 'best': best, 'wait': wait, 'first_grad': first_grad,
                          'elapsed': elapsed + time.perf_counter() - started, 'rng': rng_state(device)}, folder / 'latest.pt')
        print(f"DENSE {spec['name']} seed={seed} budget={budget:g} round={round_index+1} "
              f"source-selection pixel AP={ap:.4f} seconds={history[-1]['seconds']:.1f}", flush=True)
    restore_trainable(model, torch.load(folder / 'best.pt', map_location='cpu', weights_only=True))
    logits, local = native_evaluation(model, validation, device, batch, folder)
    selected_logits, selected_scores, _ = selection_predictions(model, validation, device, batch)
    # Fresh architecture + immutable pretrained buffers + saved trainable parameters.
    # This detects missing heads/adapters and unintended BN-buffer changes.
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_count = sum(p.numel() for p in model.parameters())
    del model, optimizer; gc.collect()
    if device.type == 'mps': torch.mps.empty_cache()
    model, identity_again, _ = build_system(spec, seed, settings['input_size'], device, bank=bank)
    if identity != identity_again: raise RuntimeError('Encoder identity changed at reload')
    restore_trainable(model, torch.load(folder / 'best.pt', map_location='cpu', weights_only=True))
    reloaded_logits, reloaded_scores, _ = selection_predictions(model, validation, device, batch)
    np.testing.assert_allclose(selected_logits, reloaded_logits, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(selected_scores, reloaded_scores, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(logits, reloaded_logits, atol=2e-5, rtol=2e-5)
    threshold = recall_threshold(y, logits, settings['target_source_recall'])
    temperature = calibration(y, logits)
    result = {**signature, 'completed_at': now(), 'training_images': len(selected), 'validation_images': len(validation),
              'total_pixel_annotated_images_including_validation': len(selected) + len(validation),
              'trainable_parameters': trainable_count, 'total_parameters': parameter_count,
              'first_gradient_check': first_grad, 'source_logit_threshold': threshold, 'source_temperature': temperature,
              'validation_metrics': classification(y, logits, threshold), 'validation_localization': local,
              'validation_calibration_raw': probability_metrics(y, logits),
              'validation_calibration_temperature_scaled': probability_metrics(y, logits, temperature),
              'best_round': max(history, key=lambda x: x['validation_selection_pixel_AP'])['round'],
              'completed_rounds': len(history), 'optimizer_updates': len(history) * steps,
              'training_and_validation_seconds_excluding_preparation': elapsed + time.perf_counter() - started,
              'checkpoint_sha256': sha256(folder / 'best.pt'),
              'checkpoint_reload': {'fresh_model': True, 'logit_max_abs_error': float(np.max(abs(selected_logits-reloaded_logits))),
                                    'sampled_pixel_probability_max_abs_error': float(np.max(abs(selected_scores-reloaded_scores)))},
              'selection': settings['selection'], 'supervision': 'same source image labels and canonical pixel masks; validation masks charged',
              'scope': 'Source development only; explicit local-patch or fresh-head frozen post-LoRA control',
              'local_objective': ('patch-max-preserved targets, image BCE + weighted patch CE +0.5 patch Dice' if spec['architecture']=='dinov3_local_patch' else 'same image/pixel objective as parent dense study'),
              'adapted_encoder_export': spec.get('adapted_encoder_path')}
    np.savez_compressed(folder / 'validation_predictions.npz', ids=np.array([r['id'] for r in source_validation]),
                        groups=np.array([r['split_group'] for r in source_validation]), y=y, logits=logits)
    save_json(folder / 'run.json', result); (folder / 'latest.pt').unlink(missing_ok=True)
    del model, training, validation, bank; gc.collect()
    if device.type == 'mps': torch.mps.empty_cache()
    return result
