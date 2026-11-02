"""Source-only dense supervision, bounded checkpoints and native-mask reporting."""
import gc
import math
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import ROOT, digest, sha256, now, save_json, read_json, choose_device
from .data import IGNORE
from .encoders import FrozenEncoder, reproject
from .full_models import dense_loss
from .image_classification import (atomic_torch_save, trainable_state, restore_trainable,
                                   rng_state, restore_rng, assert_source_partitions)
from .localization_metrics import pooled_native_metrics
from .metrics import classification, recall_threshold
from .publication_train import budget_rows, round_plan, calibration, probability_metrics
from .publication_dense_data import prepare_dense_rows, augmented_batch
from .publication_dense_models import make_dense_dino, PublicationDeepLabPlus
from .publication_normal_bank import OrientationNormalBank


FILES = ['publication_dense_models.py', 'publication_dense_data.py', 'publication_normal_bank.py',
         'publication_dense_train.py', 'full_models.py', 'publication_models.py', 'publication_train.py',
         'image_classification.py', 'classification_heads.py', 'adapt.py', 'data.py', 'encoders.py',
         'metrics.py', 'localization_metrics.py', 'common.py']


def scientific_hashes():
    return {name: sha256(ROOT / 'elresearch' / name) for name in FILES}


def build_system(spec, seed, size, device, training_rows=None, bank=None):
    if spec['architecture'] == 'dinov3_multidepth':
        encoder = FrozenEncoder('dinov3_small', size, str(device))
        identity = {k: v for k, v in encoder.identity.items() if k != 'device'}
        if spec.get('reference') and bank is None:
            if training_rows is None:
                raise ValueError('Reference construction requires explicit selected training rows')
            bank = OrientationNormalBank.build(training_rows, encoder.model, size, identity, device)
        model = make_dense_dino(encoder.model, spec['adaptation'], seed, bank,
                               spec.get('geometry_weight', .1), spec.get('top_fraction', .05))
    elif spec['architecture'] == 'deeplabv3plus_resnet50':
        if spec['adaptation'] != 'full' or spec.get('reference'):
            raise ValueError('CNN control is fully trained without a normal-reference bank')
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        encoder = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        path = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlparse(weights.url).path).name
        identity = {'name': 'resnet50', 'weights': str(weights), 'weights_sha256': sha256(path),
                    'output_stride': 16, 'input_size': size, 'dtype': 'float32',
                    'preprocessing': 'same whole-field grayscale RGB letterbox/ImageNet normalization'}
        model = PublicationDeepLabPlus(encoder, seed, spec.get('top_fraction', .05))
    else:
        raise ValueError(spec['architecture'])
    return model.to(device), identity, bank


def selection_predictions(model, examples, device, batch_size):
    model.eval()
    logits, scores, truth = [], [], []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            items = examples[start:start + batch_size]
            b = augmented_batch(items, [0] * len(items), device)
            out = model(b['images'], b['valid'], b['geometries'], b['groups'])
            logits.extend(out['image'].cpu().numpy())
            probability = out['segmentation'].softmax(1)[:, 1].cpu().numpy()
            for item, value in zip(items, probability):
                index = item['selection_indices']
                scores.append(value.ravel()[index])
                truth.append(item['target'].numpy().ravel()[index] == 1)
    logits, scores, truth = np.asarray(logits, np.float64), np.concatenate(scores), np.concatenate(truth)
    if not np.isfinite(logits).all() or not np.isfinite(scores).all() or not truth.any():
        raise ValueError('Invalid validation logits or no sampled crack pixels')
    return logits, scores, truth


def native_evaluation(model, examples, device, batch_size, folder=None):
    model.eval()
    items, logits = [], []
    # Fixed annotation-based examples, never ranked by model success.
    positives = sorted([x for x in examples if x['row']['target']],
                       key=lambda x: (x['row']['crack_area_fraction'], x['row']['id']))
    negatives = sorted([x for x in examples if not x['row']['target']], key=lambda x: x['row']['id'])
    selected = {positives[int(round(q * (len(positives) - 1)))]['row']['id'] for q in [0, .25, .5, .75, 1]} if positives else set()
    selected |= {x['row']['id'] for x in negatives[:3]}
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            b = augmented_batch(batch, [0] * len(batch), device)
            output = model(b['images'], b['valid'], b['geometries'], b['groups'])
            logits.extend(output['image'].cpu().numpy())
            for item, probability in zip(batch, output['segmentation'].softmax(1)[:, 1].cpu().numpy()):
                native = np.clip(reproject(probability, item['geometry']), 0, 1)
                items.append({'mask': item['native_mask'], 'score': native,
                              'normal_structure': item.get('native_structure')})
                if folder is not None and item['row']['id'] in selected:
                    destination = folder / 'fixed_validation_examples' / (digest(item['row']['id']) + '.npz')
                    destination.parent.mkdir(exist_ok=True)
                    np.savez_compressed(destination, id=np.array(item['row']['id']), score=native.astype(np.float16))
    result = pooled_native_metrics(items, .5)
    for row, item in zip(result['per_image'], examples):
        row.update(id=item['row']['id'], acquisition_family=item['row']['split_group'])
    result['fixed_example_ids'] = sorted(selected)
    return np.asarray(logits, np.float64), result


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
    training = prepare_dense_rows(selected, settings['input_size'])
    validation = prepare_dense_rows(source_validation, settings['input_size'])
    counts = sum((torch.bincount(x['target'][x['target'] != IGNORE], minlength=3) for x in training), torch.zeros(3, dtype=torch.long))
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
            b = augmented_batch([training[int(i)] for i in order[k]], views[k], device)
            factor = .1 + .9 * (1 + math.cos(math.pi * (round_index * steps + update) / total_steps)) / 2
            for group, base in zip(optimizer.param_groups, [spec['encoder_lr'], spec['head_lr']]):
                group['lr'] = base * factor
            output = model(b['images'], b['valid'], b['geometries'], b['groups'])
            loss = dense_loss(output, b['target'], b['y'], weights)
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
              'scope': 'Source development only; adapted DeepLabv3+ baseline is not PV-S3 original-method replication'}
    np.savez_compressed(folder / 'validation_predictions.npz', ids=np.array([r['id'] for r in source_validation]),
                        groups=np.array([r['split_group'] for r in source_validation]), y=y, logits=logits)
    save_json(folder / 'run.json', result); (folder / 'latest.pt').unlink(missing_ok=True)
    del model, training, validation, bank; gc.collect()
    if device.type == 'mps': torch.mps.empty_cache()
    return result
