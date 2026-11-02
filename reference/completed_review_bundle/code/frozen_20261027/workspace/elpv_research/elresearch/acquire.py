"""Pinned acquisition. Downloads are resumable; no test pixels are modeled."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import quote

import requests

from .common import ROOT, save_json, read_json, now

REPOS = {'ucf': 'ucf-photovoltaics/UCF-EL-Defect',
         'scdd': 'TheMakiran/BenchmarkELimages'}

def get_json(url):
    response = requests.get(url, headers={'User-Agent': 'ELPV-research/1.0'}, timeout=60)
    response.raise_for_status()
    return response.json()

def get_subtree(repo, commit, parts):
    tree_id = commit
    for component in parts:
        tree = get_json(f'https://api.github.com/repos/{repo}/git/trees/{tree_id}')
        tree_id = next(x['sha'] for x in tree['tree'] if x['path'] == component)
    result = get_json(f'https://api.github.com/repos/{repo}/git/trees/{tree_id}?recursive=1')
    if result.get('truncated'):
        raise RuntimeError('Refusing incomplete GitHub file inventory')
    prefix = '/'.join(parts)
    return [{**x, 'path': prefix + '/' + x['path']} for x in result['tree'] if x['type'] == 'blob']

def inventory(dataset):
    dest = ROOT / 'data' / dataset / 'inventory.json'
    if dest.exists() and read_json(dest).get('pairs'):
        return read_json(dest)
    repo = REPOS[dataset]
    info = get_json(f'https://api.github.com/repos/{repo}')
    commit = get_json(f"https://api.github.com/repos/{repo}/commits/{info['default_branch']}")['sha']
    if dataset == 'ucf':
        # Walking only training avoids the large duplicated preview directories.
        files = get_subtree(repo, commit, ['training'])
        image_paths = {x['path']: x for x in files if '/images/' in x['path']}
        pairs = []
        paths = {x['path'] for x in files}
        for path, entry in sorted(image_paths.items()):
            name = Path(path).name
            mask = path.replace('/images/', '/annotations/')
            if mask not in paths:
                candidates = [str(Path(mask).with_suffix(s)) for s in ['.png', '.jpg']]
                candidates += [str(Path(mask).parent / ('Annotated_' + Path(mask).stem + '.png'))]
                mask = next((m for m in candidates if m in paths), None)
            if not mask or not name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            group = re.sub(r'[_ -]?[cC]\d+(?=\.[^.]+$)', '', name)
            group = str(Path(group).with_suffix(''))
            # Conservative family grouping joins repeated pressure/time captures.
            family = re.split(r'(?:\d{8}_\d{6}|\b\d+(?:\.\d+)?\s*(?:Pa|kPa)\b)', group)[0].strip(' _-')
            simulated = bool(re.search(r'simul|synthe|generated|\bfake\b', name, re.I))
            pairs.append({'id': 'ucf/' + name, 'name': name, 'image_source': path,
                          'mask_source': mask, 'origin_id': name,
                          'capture_id': group, 'group_id': family or group,
                          'module_id': None, 'real_or_synthetic': 'synthetic' if simulated else 'unverified',
                          'upstream_split': 'train' if '/train/' in path else 'test',
                          'bytes': entry.get('size', 0)})
    else:
        files = get_subtree(repo, commit, ['dataset_20221008'])
        paths = {x['path'] for x in files}
        pairs = []
        for entry in files:
            path = entry['path']
            if not re.search(r'/el_images_(train|val|test)/', path):
                continue
            name = Path(path).name
            if name.startswith(('flip_', 'mirror_', 'rotate_')):
                continue
            mask = path.replace('/el_images_', '/el_masks_')
            if mask not in paths:
                continue
            group = re.sub(r'(?:_r\d+_c\d+|_cell\d+|[-_]cell\d+).*$', '', Path(name).stem)
            pairs.append({'id': 'scdd/' + name, 'name': name, 'image_source': path,
                          'mask_source': mask, 'origin_id': name, 'capture_id': group,
                          'group_id': group, 'module_id': None, 'real_or_synthetic': 'real',
                          'upstream_split': re.search(r'el_images_(\w+)/', path)[1],
                          'quarantine': name.startswith(('BMRK_', 'SDLE_')),
                          'bytes': entry.get('size', 0)})
    seen = set()
    unique = []
    for pair in pairs:
        if pair['id'] not in seen:
            seen.add(pair['id']); unique.append(pair)
    output = {'dataset': dataset, 'repo': repo, 'commit': commit, 'created_at': now(),
              'license': info.get('license'), 'pairs': unique,
              'limitations': ['Filename-based conservative acquisition groups are not confirmed physical module IDs.',
                             'UCF real/synthetic provenance requires upstream documentation; unverified files are not publication-eligible.']}
    save_json(dest, output)
    print(f'{dataset}: pinned {commit}; {len(unique)} original image/mask pairs', flush=True)
    if not unique:
        raise RuntimeError('No paired images found; acquisition requires a valid inventory')
    return output

def download_file(repo, commit, remote, local):
    local = Path(local)
    if local.exists() and local.stat().st_size:
        return
    url = f'https://raw.githubusercontent.com/{repo}/{commit}/' + quote(remote, safe='/')
    local.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            content = response.content
            if content.startswith(b'version https://git-lfs.github.com/spec'):
                response = requests.get(f'https://media.githubusercontent.com/media/{repo}/{commit}/' + quote(remote, safe='/'), timeout=120)
                response.raise_for_status(); content = response.content
            if not content or content.startswith((b'<!DOCTYPE', b'<html')):
                raise ValueError('Response is not a data file')
            tmp = local.with_suffix(local.suffix + '.part')
            tmp.write_bytes(content); tmp.replace(local)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)

def acquire(dataset='ucf', limit=768, workers=8):
    inv = inventory(dataset)
    pairs = [p for p in inv['pairs'] if p['real_or_synthetic'] != 'synthetic' and not p.get('quarantine')]
    # Label-blind fixed selection; no selection based on model scores or outcome.
    pairs = sorted(pairs, key=lambda p: hashlib.sha256(p['id'].encode()).hexdigest())
    if limit:
        pairs = pairs[:limit]
    root = ROOT / 'data' / dataset
    selection = root / f'selection_{limit or "all"}.json'
    output = {'commit': inv['commit'], 'dataset': dataset, 'selection': 'label-blind sha256 order',
              'publication_eligible': dataset != 'ucf', 'pairs': pairs}
    save_json(selection, output)
    def task(pair):
        for kind in ['image', 'mask']:
            suffix = Path(pair[kind + '_source']).suffix
            path = root / 'raw' / kind / (hashlib.sha256(pair['id'].encode()).hexdigest() + suffix)
            download_file(inv['repo'], inv['commit'], pair[kind + '_source'], path)
            pair[kind + '_path'] = str(path.relative_to(ROOT))
        return pair
    finished, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, p.copy()): p for p in pairs}
        for future in concurrent.futures.as_completed(futures):
            try:
                finished.append(future.result())
            except Exception as e:
                failures.append({'id': futures[future]['id'], 'error': str(e)})
            if (len(finished) + len(failures)) % 50 == 0:
                print(f'{dataset}: {len(finished)}/{len(pairs)} pairs downloaded; errors={len(failures)}', flush=True)
                save_json(root / 'download_status.json', {'complete': len(finished), 'requested': len(pairs), 'failures': failures, 'updated_at': now()})
    output.update(pairs=sorted(finished, key=lambda p: p['id']), failures=failures, download_complete=not failures)
    save_json(root / 'downloaded.json', output)
    save_json(root / 'download_status.json', {'complete': len(finished), 'requested': len(pairs), 'failures': failures, 'updated_at': now()})
    return output
