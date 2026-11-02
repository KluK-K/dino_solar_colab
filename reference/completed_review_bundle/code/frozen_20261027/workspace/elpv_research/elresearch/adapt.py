"""Source-only DINOv3 Q/V LoRA; export a merged frozen encoder for fresh caches."""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, config, digest, read_json, save_json, sha256, now
from .data import records, load_image
from .encoders import image_tensor, patch_geometry, split_tokens
from .heads import budget_rows
from .metrics import classification, recall_threshold

class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        if rank < 1:
            raise ValueError('LoRA rank must be positive')
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale

    def merged(self):
        with torch.no_grad():
            self.base.weight.add_((self.b @ self.a) * self.scale)
        return self.base.requires_grad_(False)

def inject_lora(model, rank=8, blocks=4):
    layers = getattr(model, 'layer', None)
    if layers is None or len(layers) < blocks:
        raise ValueError('Expected official DINOv3 ViT layer/attention/q_proj/v_proj architecture')
    model.requires_grad_(False)
    for layer in layers[-blocks:]:
        for name in ['q_proj', 'v_proj']:
            base = getattr(layer.attention, name)
            if not isinstance(base, nn.Linear):
                raise ValueError('Projection already adapted or architecture incompatible')
            setattr(layer.attention, name, LoRALinear(base, rank, rank))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def merge_lora(model):
    for layer in model.layer:
        for name in ['q_proj', 'v_proj']:
            module = getattr(layer.attention, name)
            if isinstance(module, LoRALinear):
                setattr(layer.attention, name, module.merged())
    model.eval().requires_grad_(False)

def adapt_lora(dataset, encoder, budget=1., seed=11, epochs=40, batch_size=4):
    if encoder.name != 'dinov3_small' or dataset == 'elpv':
        raise ValueError('Initial LoRA protocol is DINOv3 Small on a crack-mask dataset')
    cfg = config(); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    train = budget_rows(records(dataset, 'train'), budget, seed)
    validation = records(dataset, 'validation')
    manifest = read_json(ROOT / 'data' / dataset / 'manifest.json')
    signature = {'dataset': dataset, 'source_manifest': manifest['signature'], 'encoder': encoder.identity,
                 'seed': seed, 'budget': budget, 'rank': 8, 'last_blocks': 4, 'alpha': 8,
                 'epochs': epochs, 'batch_size': batch_size, 'adapter_lr': 1e-4, 'head_lr': 1e-3,
                 'weight_decay': cfg['weight_decay'], 'patience': cfg['patience'], 'code': sha256(__file__)}
    run_id = digest(signature)[:20]; folder = ROOT / 'artifacts/adaptations' / run_id
    if (folder / 'run.json').exists():
        raise RuntimeError('Completed adaptation is immutable; reuse its exported encoder')
    folder.mkdir(parents=True, exist_ok=True)
    trainable = inject_lora(encoder.model)
    encoder.model.eval()  # Keep pretrained stochastic layers disabled; adapters still receive gradients.
    head = nn.Linear(encoder.model.config.hidden_size * 2, 1).to(encoder.device)
    optimizer = torch.optim.AdamW([{'params': [p for p in encoder.model.parameters() if p.requires_grad], 'lr': 1e-4},
                                  {'params': head.parameters(), 'lr': 1e-3}], weight_decay=cfg['weight_decay'])

    def prepare(rows):
        examples = []
        for row in rows:
            tensor, g = image_tensor(load_image(row), encoder.size)
            valid, _ = patch_geometry(g, encoder.patch_size)
            examples.append((tensor, torch.from_numpy(valid), row['target']))
        return examples

    train_examples, val_examples = prepare(train), prepare(validation)
    def logits(examples):
        image = torch.stack([x[0] for x in examples]).to(encoder.device)
        keep = torch.stack([x[1] for x in examples]).to(encoder.device)
        hidden = encoder.model(pixel_values=image).last_hidden_state
        cls, patches = split_tokens(hidden, encoder.size // encoder.patch_size, encoder.model.config.num_register_tokens)
        mean = (patches * keep[..., None]).sum((1, 2)) / keep.sum((1, 2))[:, None]
        return head(torch.cat([cls, mean], -1)).squeeze(-1)

    def predict():
        with torch.no_grad():
            return np.concatenate([logits(val_examples[start:start+batch_size]).sigmoid().cpu().numpy()
                                   for start in range(0, len(val_examples), batch_size)])

    y = np.array([r['target'] for r in validation]); history, best, wait = [], -np.inf, 0
    for epoch in range(epochs):
        losses = []
        permutation = rng.permutation(len(train_examples))
        for start in range(0, len(permutation), batch_size):
            selected = [train_examples[i] for i in permutation[start:start+batch_size]]
            output = logits(selected)
            target = torch.tensor([x[2] for x in selected], dtype=torch.float32, device=encoder.device)
            loss = F.binary_cross_entropy_with_logits(output, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        scores = predict(); auc = classification(y, scores, .5)['auc']
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'validation_auc': auc})
        save_json(folder / 'history.json', history)
        print(f'LoRA epoch {epoch+1}: loss={np.mean(losses):.4f}, validation AUC={auc:.4f}', flush=True)
        if auc > best:
            best, wait = auc, 0
            torch.save({'adapter': {n: p.detach().cpu() for n, p in encoder.model.named_parameters() if p.requires_grad},
                        'head': {n: p.detach().cpu() for n, p in head.state_dict().items()}}, folder / 'adapter.pt')
        else:
            wait += 1
            if wait >= cfg['patience']:
                break
    state = torch.load(folder / 'adapter.pt', map_location=encoder.device, weights_only=True)
    encoder.model.load_state_dict(state['adapter'], strict=False); head.load_state_dict(state['head'])
    scores = predict(); threshold = recall_threshold(y, scores, cfg['review_recall'])
    merge_lora(encoder.model)
    destination = folder / 'backbone'; encoder.model.save_pretrained(destination, safe_serialization=True)
    result = {'run_id': run_id, 'kind': 'adaptation', **signature, 'created_at': now(),
              'training_ids': [r['id'] for r in train], 'validation_ids': [r['id'] for r in validation],
              'trainable_encoder_parameters': trainable, 'exported_backbone': str(destination),
              'merged_weights_sha256': sha256(destination / 'model.safetensors'),
              'validation_metrics': classification(y, scores, threshold),
              'publication_eligible': manifest.get('publication_eligible', False),
              'supervision': 'source image crack labels; masks are not used by the adaptation objective',
              'next_step': 'Load exported backbone via --model-path; re-extract features and rebuild source normal bank. Freeze the resulting probe/head run before evaluation.'}
    save_json(folder / 'run.json', result)
    np.savez_compressed(folder / 'validation_predictions.npz', ids=np.array([r['id'] for r in validation]), y=y, scores=scores)
    return result
