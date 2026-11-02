"""Hash-guarded editorial revision of SAME v04 with complete evidence retention.

Patch JSON contains exact old/new text replacements and explicit new evidence
files. Historical embedded PDFs and every bound_original category are retained.
This utility never edits scientific code/registries, runs models or claims QA.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import re
import shutil

ROOT = Path(__file__).absolute().parent
BASE = ROOT / 'artifacts/publication_20261017'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ref(p):
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def read(p):
    return json.loads(p.read_text())


def save(p, d):
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('patch', type=Path); args = parser.parse_args()
    patch_path = args.patch.absolute(); patch = read(patch_path)
    source = ROOT / 'manuscript/research_manuscript_v04.tex'; metadata = ROOT / 'manuscript/manuscript_v04_inputs.json'
    meta = read(metadata); before = source.read_text()
    old_revision = meta['revision']; revision = old_revision + 1
    assert patch['expected_revision'] == old_revision and patch['expected_source_sha256'] == sha(source) == meta['manuscript_sha256']
    assert re.fullmatch('[a-z0-9_]+', patch['slug'])
    frozen = BASE / 'final_confirmation_v1/frozen_registry.json'
    assert sha(frozen) == 'eb28c73d11482ad0881600a1e7d3fedee826f66f09b2e96e98f7de222881eb3f'
    registry = read(frozen)
    for b in registry['evaluator_files']: assert sha(ROOT/b['path']) == b['sha256']
    text = before
    for replacement in patch['replacements']:
        assert text.count(replacement['old']) == 1, replacement['old'][:120]
        text = text.replace(replacement['old'], replacement['new'], 1)
    date = f'article draft v04, revision {old_revision}'
    assert text.count(date) == 1
    text = text.replace(date, f'article draft v04, revision {revision}', 1)
    assert all(block in text for block in re.findall(r'\\begin\{filecontents\*\}.*?\\end\{filecontents\*\}', before, re.S))
    assert r'\author{}' in text
    old_manifest = ROOT / meta['evidence_manifest']['path']; assert sha(old_manifest) == meta['evidence_manifest']['sha256']
    old_evidence = read(old_manifest)
    needed = sum((old_manifest.parent/x['archive_file']).stat().st_size for x in old_evidence['copied_sources'])
    assert shutil.disk_usage(ROOT).free - needed > 20*2**30
    at = datetime.now(timezone.utc).isoformat()
    proof_dir = BASE / 'manuscript_editorial_revisions_v1' / at.replace(':','-'); proof_dir.mkdir(parents=True)
    shutil.copy2(__file__, proof_dir/Path(__file__).name); shutil.copy2(patch_path, proof_dir/'applied_patch.json')
    prior = ROOT/'manuscript/revisions'/(sha(source)+'__'+patch['slug']+'_'+sha(metadata)[:12]); prior.mkdir()
    for p in [source, metadata]: shutil.copy2(p,prior/p.name)
    archive = ROOT/'manuscript/evidence_v04'/f'revision{revision}_{patch["slug"]}'; archive.mkdir()
    copied = []

    def capture(p, name):
        assert p.is_file() and not p.is_symlink(); digest = sha(p); dest = archive/name
        assert not dest.exists(); shutil.copy2(p,dest); assert sha(dest) == digest
        copied.append(dict(source=os.path.relpath(p,ROOT),source_sha256=digest,archive_file=name,archive_sha256=digest))

    for i,row in enumerate(old_evidence['copied_sources']):
        p=old_manifest.parent/row['archive_file']; assert sha(p)==row['archive_sha256']
        capture(p,f'prior{old_revision}__{i:05d}__{sha(p)[:12]}__'+p.name[-96:])
    capture(old_manifest,f'prior{old_revision}__evidence_manifest.json')
    extras = [patch_path, Path(__file__).absolute(), proof_dir/'applied_patch.json', proof_dir/Path(__file__).name]
    for name in patch.get('evidence_files',[]):
        p=ROOT/name; assert p.absolute().is_relative_to(ROOT)
        extras.append(p)
    for i,p in enumerate(dict.fromkeys(extras)):
        capture(p,f'new{revision}__{i:05d}__{sha(p)[:12]}__'+p.name[-96:])
    retained={k:v for k,v in old_evidence.items() if k.startswith('bound_original_')}
    manifest=archive/'evidence_manifest.json'
    save(manifest,dict(at=at,revision=revision,copied_sources=copied,**retained,
        unique_archive_names=len(copied),previous_manifest=meta['evidence_manifest'],scope=patch['scope']))
    for b in registry['evaluator_files']: assert sha(ROOT/b['path'])==b['sha256']
    assert sha(frozen)=='eb28c73d11482ad0881600a1e7d3fedee826f66f09b2e96e98f7de222881eb3f'
    temp=source.with_suffix('.tex.atomic_edit');temp.write_text(text);os.replace(temp,source)
    meta.update(at=at,revision=revision,manuscript_sha256=sha(source),previous_revision_directory=str(prior.relative_to(ROOT)),
                archived_evidence_directory=str(archive.relative_to(ROOT)),evidence_manifest=ref(manifest),
                latest_editorial_patch=ref(proof_dir/'applied_patch.json'),
                native_compiler=dict(status='pending_current_revision',source_sha256=sha(source),current_pdf_export_verified=False,current_pdf_page_QA=False))
    for k,v in patch.get('metadata_updates',{}).items():
        assert k not in ['revision','manuscript_sha256','evidence_manifest','native_compiler','submission_ready']
        meta[k]=v
    save(metadata,meta)
    save(proof_dir/'verification.json',dict(at=at,revision=revision,source=ref(source),evidence_manifest=ref(manifest),
        archived_input_files=len(copied),all_bound_original_categories_retained={k:len(v) for k,v in retained.items()},
        existing_embedded_PDFs_retained=True,all92scientific_files_and_registry_unchanged=True,
        native_compile_export_page_QA_pending=True,submission_ready=False))
    print(json.dumps(dict(revision=revision,source_sha256=sha(source),archived_inputs=len(copied),proof=str(proof_dir)),indent=2))


if __name__=='__main__':main()
