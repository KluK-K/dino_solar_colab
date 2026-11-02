"""Standard-library byte/AST/exclusion check only; never imports model code."""
from pathlib import Path
import ast,hashlib,json,re,sys,warnings
root=Path(__file__).resolve().parent
m=json.loads((root/'CODE_MANIFEST.json').read_text())
expected={x['path']:x for x in m['files']}
actual={str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file() and p.name!='CODE_MANIFEST.json'}
assert set(actual)==set(expected), 'Missing or unlisted code snapshot files'
bad_suffix={'.pt','.pth','.safetensors','.npz','.npy','.joblib','.pkl','.zip','.gz','.tar','.ckpt','.pyc'}
bad_parts={'.venv','environment','data','cache','models','__pycache__','PV-S3_upstream_code','PS-MT_upstream_code'}
secrets=re.compile(r'(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
parsed=0;notes=[]
for rel,p in actual.items():
    assert not p.is_symlink() and not (set(Path(rel).parts)&bad_parts) and p.suffix not in bad_suffix, rel
    raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected[rel]['sha256'] and len(raw)==expected[rel]['bytes'],rel
    if p.suffix in {'.py','.txt','.md','.json','.ini'}:
        assert not secrets.search(raw.decode()), 'Credential pattern in snapshot'
    if p.suffix=='.py':
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');ast.parse(raw.decode(),filename=rel)
        notes.extend({'file':rel,'line':w.lineno,'message':str(w.message)} for w in caught);parsed+=1
context=json.loads((root/'FROZEN_CONTEXT.json').read_text())
assert context['registry_id']==m['frozen_registry_id']
for binding in m['frozen_python_closure']:
    assert expected[binding['snapshot_path']]['sha256']==binding['sha256']
assert len(m['frozen_python_closure'])==56
config=json.loads((root/'workspace/elpv_research/config.json').read_text())
assert 'dinov2_small' in config['disabled_encoders']
oracle=json.loads((root/'workspace/elpv_research/references/verification_sources_manifest_v1.json').read_text())
assert len(oracle['files'])==4
assert not any('upstream_code' in Path(p).parts for p in actual)
print(json.dumps({'passed':True,'files_checked':len(actual),'Python_sources_AST_checked':parsed,
    'frozen_python_closure_files_exact':56,'author_oracle_recipe_files':4,'unverified_licensed_oracle_source_bundled':False,
    'source_parse_warnings':notes,'network_model_test_training_or_inference_performed':False,
    'scope':'Byte/AST/exclusion and frozen-code closure checks only. No implementation-suite, authorized-input, research-performance or cross-device reproduction claim.'},indent=2))
