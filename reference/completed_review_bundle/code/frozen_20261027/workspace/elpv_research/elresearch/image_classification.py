"""Image-label-only head/adaptation study. Never loads a segmentation mask or test split."""
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .adapt import inject_lora
from .classification_heads import make_image_head, head_spec
from .common import ROOT, choose_device, digest, now, read_json, save_json, sha256
from .data import records, load_image
from .encoders import FrozenEncoder, cache_path, extract, image_tensor
from .metrics import classification, recall_threshold


SCIENTIFIC_FILES = ['classification_heads.py', 'image_classification.py', 'adapt.py',
                    'data.py', 'encoders.py', 'metrics.py', 'common.py']


def scientific_hashes():
    return {name: sha256(ROOT / 'elresearch' / name) for name in SCIENTIFIC_FILES}


class EmbeddingClassifier(nn.Module):
    def __init__(self, backbone, dimensions, encoder_name='dinov3_small',
                 adaptation='frozen', head='linear', seed=11):
        super().__init__()
        if adaptation not in {'frozen', 'lora', 'full'}:
            raise ValueError(adaptation)
        if encoder_name != 'dinov3_small' and adaptation != 'frozen':
            raise ValueError('CNN controls in this study use frozen ImageNet features')
        self.encoder, self.encoder_name, self.adaptation = backbone, encoder_name, adaptation
        self.encoder.requires_grad_(False)
        # Independent streams keep head initialization paired across adaptation modes.
        if adaptation == 'lora':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 10000)
                inject_lora(self.encoder, rank=8, blocks=4)
        elif adaptation == 'full':
            self.encoder.requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = make_image_head(dimensions, head)
        self.dimensions = dimensions

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Same deterministic pretrained behavior in every arm.
        return self

    def classify(self, embeddings):
        # Per-image normalization: no fitted source or target statistics.
        value = F.layer_norm(embeddings, (self.dimensions,))
        return self.head(value).squeeze(-1)

    def forward(self, images):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.adaptation != 'frozen'):
            if self.encoder_name == 'dinov3_small':
                embeddings = self.encoder(pixel_values=images).last_hidden_state[:, 0]
            else:
                embeddings = self.encoder(images)
        return self.classify(embeddings)


def trainable_state(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name in names}


def restore_trainable(model, state):
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError('Checkpoint does not match the trainable-parameter contract')
    model.load_state_dict(state, strict=False)


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def cached_embeddings(encoder, rows):
    values = []
    for row in rows:
        with np.load(cache_path(encoder, row), allow_pickle=False) as item:
            values.append(item['cls'].astype(np.float32))
    return torch.from_numpy(np.stack(values))


def prepare_images(rows, size):
    return torch.stack([image_tensor(load_image(row), size)[0] for row in rows])


def rng_state(device):
    value = {'cpu': torch.get_rng_state(), 'python': random.getstate()}
    if device.type == 'mps':
        value['mps'] = torch.mps.get_rng_state()
    elif device.type == 'cuda':
        value['cuda'] = torch.cuda.get_rng_state_all()
    return value


def restore_rng(value, device):
    torch.set_rng_state(value['cpu']); random.setstate(value['python'])
    if device.type == 'mps':
        torch.mps.set_rng_state(value['mps'])
    elif device.type == 'cuda':
        torch.cuda.set_rng_state_all(value['cuda'])


def train_image_classifier(study, folder, specification, seed):
    """Completed runs are immutable; interrupted epochs resume from atomic checkpoints."""
    folder = Path(folder)
    if (folder / 'run.json').exists():
        result = read_json(folder / 'run.json')
        if sha256(folder / 'model.pt') != result['checkpoint_sha256']:
            raise RuntimeError('Completed classifier checkpoint changed')
        return result
    if scientific_hashes() != study['scientific_hashes']:
        raise RuntimeError('Scientific code changed after protocol registration')
    manifest = read_json(ROOT / 'data' / study['dataset'] / 'manifest.json')
    if manifest['signature'] != study['source_manifest']:
        raise RuntimeError('Source cohort changed')
    train, validation = records(study['dataset'], 'train'), records(study['dataset'], 'validation')
    assert_source_partitions(train, validation)
    seed = int(seed)
    random.seed(seed); torch.manual_seed(seed); order_rng = np.random.default_rng(seed)
    requested_device = choose_device(study['device'])
    encoder = FrozenEncoder(specification['encoder'], study['input_size'], str(requested_device))
    dimensions = 384 if specification['encoder'] == 'dinov3_small' else 512
    cached = specification['adaptation'] == 'frozen'
    if cached:
        extract(study['dataset'], encoder, 'train', study['batch_size'])
        extract(study['dataset'], encoder, 'validation', study['batch_size'])
        train_x, val_x = cached_embeddings(encoder, train), cached_embeddings(encoder, validation)
        # Small head-only optimization is quicker on CPU; the encoder is identical.
        device = torch.device('cpu')
        encoder.model.to('cpu')
    else:
        train_x, val_x = prepare_images(train, study['input_size']), prepare_images(validation, study['input_size'])
        device = requested_device
    model = EmbeddingClassifier(encoder.model, dimensions, specification['encoder'],
                                specification['adaptation'], specification['head'], seed).to(device)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': study['encoder_lr']},
        {'params': model.head.parameters(), 'lr': study['head_lr']}], weight_decay=study['weight_decay'])
    train_y = torch.tensor([r['target'] for r in train], dtype=torch.float32)
    val_y = np.array([r['target'] for r in validation])
    signature = {'study_id': study['study_id'], 'specification': specification, 'seed': seed,
                 'encoder': encoder.identity, 'head': head_spec(dimensions, specification['head']),
                 'training_ids': [r['id'] for r in train], 'validation_ids': [r['id'] for r in validation]}
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'started.json').exists():
        # JSON converts integer config keys (e.g. id2label) into strings.
        if digest(read_json(folder / 'started.json')['signature']) != digest(signature):
            raise RuntimeError('Partial run identity changed')
    else:
        save_json(folder / 'started.json', {'started_at': now(), 'signature': signature})
    history, best, wait, elapsed, start_epoch, first_gradient = [], -float('inf'), 0, 0., 0, None
    if (folder / 'latest.pt').exists():
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        restore_trainable(model, state['model']); optimizer.load_state_dict(state['optimizer'])
        history, best, wait, elapsed = state['history'], state['best'], state['wait'], state['elapsed']
        start_epoch = len(history); first_gradient = state['first_gradient']
        order_rng.bit_generator.state = state['order_rng']; restore_rng(state['torch_rng'], device)
    started = time.perf_counter()
    batch_size = study['batch_size']

    def forward(x):
        return model.classify(x.to(device)) if cached else model(x.to(device))

    def predict(x):
        model.eval()
        with torch.inference_mode():
            return np.concatenate([forward(x[i:i+batch_size]).sigmoid().cpu().numpy()
                                   for i in range(0, len(x), batch_size)])

    for epoch in range(start_epoch, study['epochs']):
        if wait >= study['patience']:
            break
        model.train(); losses, epoch_start = [], time.perf_counter()
        order = order_rng.permutation(len(train))
        for i in range(0, len(order), batch_size):
            index = order[i:i+batch_size]
            output = forward(train_x[index]); target = train_y[index].to(device)
            loss = F.binary_cross_entropy_with_logits(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite classification loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if first_gradient is None:
                first_gradient = {
                    'encoder_parameters_with_gradients': sum(p.numel() for p in model.encoder.parameters() if p.grad is not None),
                    'encoder_gradient_l2': float(sum(p.grad.detach().square().sum().cpu() for p in model.encoder.parameters() if p.grad is not None)**.5),
                    'head_gradient_l2': float(sum(p.grad.detach().square().sum().cpu() for p in model.head.parameters() if p.grad is not None)**.5)}
                if first_gradient['head_gradient_l2'] <= 0:
                    raise RuntimeError('Classification head did not receive gradients')
                if not cached and first_gradient['encoder_gradient_l2'] <= 0:
                    raise RuntimeError('Adapted encoder did not receive gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            losses.append((float(loss.detach().cpu()), len(index)))
        scores = predict(val_x)
        auc = classification(val_y, scores, .5)['auc']
        history.append({'epoch': epoch+1,
                        'loss': sum(v*n for v, n in losses)/len(train),
                        'validation_auc': auc, 'seconds': time.perf_counter()-epoch_start})
        if auc > best:
            best, wait = auc, 0
            atomic_torch_save(trainable_state(model), folder / 'model.pt')
        else:
            wait += 1
        save_json(folder / 'history.json', history)
        atomic_torch_save({'model': trainable_state(model), 'optimizer': optimizer.state_dict(),
                          'history': history, 'best': best, 'wait': wait, 'first_gradient': first_gradient,
                          'elapsed': elapsed+time.perf_counter()-started,
                          'order_rng': order_rng.bit_generator.state, 'torch_rng': rng_state(device)},
                         folder / 'latest.pt')
        print(f"{specification['name']} seed={seed} epoch={epoch+1}: "
              f"loss={history[-1]['loss']:.4f}, val AUC={auc:.4f}, "
              f"seconds={history[-1]['seconds']:.1f}", flush=True)
    restore_trainable(model, torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
    scores = predict(val_x)
    threshold = recall_threshold(val_y, scores, study['review_recall'])
    result = {**signature, 'completed_at': now(), 'kind': 'image_only_classifier',
              'implementation_hashes': study['scientific_hashes'],
              'source_manifest': manifest['signature'], 'train_images': len(train), 'validation_images': len(validation),
              'validation_metrics': classification(val_y, scores, threshold),
              'validation_metrics_at_0_5': classification(val_y, scores, .5),
              'threshold_origin': 'same source validation; descriptive, not independent calibration',
              'best_epoch': max(history, key=lambda row: row['validation_auc'])['epoch'],
              'epochs_completed': len(history), 'optimization_seconds': elapsed+time.perf_counter()-started,
              'head_parameters': sum(p.numel() for p in model.head.parameters()),
              'trainable_encoder_parameters': sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'first_step_gradient_check': first_gradient, 'encoder_device': str(requested_device),
              'optimization_device': str(device), 'cached_embeddings': cached,
              'supervision': 'image crack-presence labels only; no pixel loss and no mask loading',
              'scope': 'exploratory source-validation comparison; not a paper reproduction or held-out result',
              'publication_eligible': manifest.get('publication_eligible', False),
              'checkpoint_sha256': sha256(folder / 'model.pt')}
    np.savez_compressed(folder / 'validation_predictions.npz', ids=np.array([r['id'] for r in validation]),
                        y=val_y, scores=scores, groups=np.array([r['split_group'] for r in validation]))
    save_json(folder / 'run.json', result)
    (folder / 'latest.pt').unlink(missing_ok=True)
    print(f"COMPLETED {specification['name']} seed={seed}: AUROC={result['validation_metrics']['auc']:.4f}", flush=True)
    return result


def assert_source_partitions(train, validation):
    for rows, partition in [(train, 'train'), (validation, 'validation')]:
        if not rows or any(r['partition'] != partition for r in rows):
            raise ValueError('Unexpected source partition')
        if set(r['target'] for r in rows) != {0, 1}:
            raise ValueError('Each source partition needs both classes')
    for key in ['id', 'split_group', 'pixel_sha256', 'image_sha256']:
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError(f'Train/validation leakage: {key}')
