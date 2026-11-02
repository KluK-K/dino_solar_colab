"""Training-only, orientation-matched normal references for a frozen encoder."""
import torch
from torch.nn import functional as F

from .common import ROOT, digest, sha256, save_json, read_json
from .data import load_image, load_mask
from .encoders import image_tensor, patch_targets, patch_geometry, split_tokens
from .image_classification import atomic_torch_save
from .publication_models import d4_view
from .publication_dense_data import rotate_geometry


class OrientationNormalBank:
    def __init__(self, items, encoder_identity):
        if not items or len({x['group'] for x in items}) < 2:
            raise ValueError('Need normal references from at least two training acquisition groups')
        self.items, self.encoder_identity = items, encoder_identity
        self.cls = [F.normalize(torch.stack([x['views'][v]['cls'].float() for x in items]), dim=-1) for v in range(8)]

    def residual(self, feature, query_group=None, topk=5, geometry_weight=.1, device='cpu'):
        if not isinstance(query_group, tuple) or len(query_group) != 2:
            raise ValueError('Expected (acquisition_family, D4_view) query identity')
        group, view = query_group
        if not 0 <= int(view) < 8 or geometry_weight < 0:
            raise ValueError('Invalid orientation or geometry weight')
        eligible = [i for i, x in enumerate(self.items) if x['group'] != group]
        if not eligible:
            raise ValueError('Self/acquisition-family reference is prohibited')
        cls = F.normalize(torch.as_tensor(feature['cls'], dtype=torch.float32), dim=-1)
        rank = (self.cls[view][eligible] @ cls).topk(min(topk, len(eligible))).indices.tolist()
        chosen = [self.items[eligible[i]]['views'][view] for i in rank]
        bank = F.normalize(torch.cat([x['patch'].float() for x in chosen]).to(device), dim=-1)
        xy_bank = torch.cat([x['coords'].float() for x in chosen]).to(device)
        query = F.normalize(torch.as_tensor(feature['patch'], dtype=torch.float32).flatten(0, 1).to(device), dim=-1)
        xy = torch.as_tensor(feature['coords'], dtype=torch.float32).flatten(0, 1).to(device)
        distance = 1 - query @ bank.T
        objective = distance + geometry_weight * (xy[:, None] - xy_bank[None]).square().sum(-1)
        match = objective.argmin(1)
        residual = query - bank[match]
        matched_distance = distance[torch.arange(len(query), device=device), match, None]
        return torch.cat([query, residual, xy, matched_distance], -1).cpu().reshape(*feature['patch'].shape[:2], -1)

    @classmethod
    def build(cls, rows, encoder, size, identity, device, limit=256):
        if any(r['partition'] != 'train' for r in rows):
            raise PermissionError('Normal references must use selected training images only')
        items = []
        for row in rows:
            key = digest({'encoder': identity, 'input_size': size,
                          'image': row['image_sha256'], 'mask': row['mask_sha256'],
                          'id': row['id'], 'bank_code': sha256(__file__),
                          'transform_code': sha256(ROOT / 'elresearch/publication_dense_data.py')})
            path = ROOT / 'cache/publication_normal_d4' / (key + '.pt')
            if path.exists():
                if sha256(path) != read_json(path.with_suffix('.json'))['payload_sha256']:
                    raise RuntimeError('Normal-bank cached tensor payload changed')
                item = torch.load(path, map_location='cpu', weights_only=True)
                if item['id'] != row['id'] or item['cache_key'] != key:
                    raise RuntimeError('Normal-bank cache identity changed')
            else:
                # Hash-checked original image/mask, never a validation/test reference.
                image, g = image_tensor(load_image(row), size)
                target = torch.from_numpy(patch_targets(load_mask(row), g, 16))
                views = []
                encoder.eval()
                with torch.inference_mode():
                    for view in range(8):
                        hidden = encoder(pixel_values=d4_view(image, view)[None].to(device)).last_hidden_state
                        cls_value, patches = split_tokens(hidden, size // 16, encoder.config.num_register_tokens)
                        keep = d4_view(target, view) == 0
                        _, coords = patch_geometry(rotate_geometry(g, view), 16)
                        views.append({'cls': cls_value[0].cpu().half(),
                                      'patch': patches[0][keep.to(device)].cpu().half(),
                                      'coords': torch.from_numpy(coords)[keep].half()})
                item = {'id': row['id'], 'group': row['split_group'], 'cache_key': key,
                        'image_sha256': row['image_sha256'], 'mask_sha256': row['mask_sha256'], 'views': views}
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_torch_save(item, path)
                save_json(path.with_suffix('.json'), {'cache_key': key, 'payload_sha256': sha256(path)})
            if all(len(x['patch']) >= 10 for x in item['views']):
                items.append(item)
        if len(items) > limit:
            vectors = F.normalize(torch.stack([x['views'][0]['cls'].float() for x in items]), dim=-1)
            selected = [0]
            distance = 1 - vectors @ vectors[0]
            for _ in range(limit - 1):
                distance[selected] = -1
                index = int(distance.argmax())
                selected.append(index)
                distance = torch.minimum(distance, 1 - vectors @ vectors[index])
            items = [items[i] for i in selected]
        return cls(items, identity)

    def write_recipe(self, folder):
        # Cache rows are shared once, rather than duplicating ~GB banks per fit.
        recipe = {'encoder': self.encoder_identity, 'cache_directory': 'cache/publication_normal_d4',
                  'rows': [{**{k: x[k] for k in ['id', 'group', 'cache_key', 'image_sha256', 'mask_sha256']},
                            'payload_sha256': sha256(ROOT / 'cache/publication_normal_d4' / (x['cache_key'] + '.pt'))} for x in self.items],
                  'orientation': 'same D4 view as query; exclude complete query acquisition family',
                  'normal_eligibility': 'canonical patch label 0 only; excludes crack, other and unknown pixels'}
        save_json(folder / 'normal_bank_recipe.json', recipe)
        return recipe

    @classmethod
    def from_recipe(cls, path, training_ids):
        recipe = read_json(path)
        if not {r['id'] for r in recipe['rows']} <= set(training_ids):
            raise PermissionError('Reference recipe contains non-training identities')
        items = []
        for row in recipe['rows']:
            source = ROOT / recipe['cache_directory'] / (row['cache_key'] + '.pt')
            if sha256(source) != row['payload_sha256']:
                raise RuntimeError('Frozen reference bank changed')
            item = torch.load(source, map_location='cpu', weights_only=True)
            if any(item[k] != row[k] for k in ['id', 'group', 'cache_key', 'image_sha256', 'mask_sha256']):
                raise RuntimeError('Reference recipe identity mismatch')
            items.append(item)
        return cls(items, recipe['encoder'])
