"""Acquire only the pinned author files used by numerical-oracle checks.

This is an explicit setup step, not a test skip or an implicit model/data
download. Original third-party terms apply; no redistribution grant is added.
Existing files must match their recorded hash and are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.parse
import urllib.request


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquire(manifest_path, destination):
    manifest_path=Path(manifest_path);destination=Path(destination).resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('version')!=1 or not manifest.get('files'):
        raise ValueError('Require a versioned explicit author-source list')
    records=[];seen=set()
    for item in manifest['files']:
        relative=PurePosixPath(item['destination']);commit=item['commit'];expected=item['sha256']
        if (relative.is_absolute() or '..' in relative.parts or str(relative) in seen
            or relative.suffix not in {'.py','.md','.txt'} or not re.fullmatch('[a-f0-9]{40}',commit)
            or not re.fullmatch('[a-f0-9]{64}',expected)):
            raise ValueError('Invalid or duplicate pinned verification-source destination')
        seen.add(str(relative))
        prefix=f"https://raw.githubusercontent.com/{item['repository']}/{commit}/"
        if not item['url'].startswith(prefix):
            raise ValueError('Only the declared official pinned raw source is allowed')
        target=destination/str(relative)
        if not target.resolve().is_relative_to(destination) or target.is_symlink():
            raise ValueError('Do not traverse a verification-source destination outside this project')
        downloaded=False
        if target.exists():
            if file_sha(target)!=expected:
                raise RuntimeError('Existing verification source differs; preserve it for explicit review instead of overwriting')
        else:
            request=urllib.request.Request(item['url'],headers={'User-Agent':'EL-research-numerical-oracle-setup/1'})
            with urllib.request.urlopen(request,timeout=30) as response:raw=response.read()
            if hashlib.sha256(raw).hexdigest()!=expected:
                raise RuntimeError('Author source bytes differ from the reviewed pinned oracle')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as output:output.write(raw)
            downloaded=True
        records.append({'destination':str(relative),'url':item['url'],'sha256':file_sha(target),
            'downloaded_this_invocation':downloaded,'license_observation':item['license_observation']})
    return {'manifest_sha256':file_sha(manifest_path),'files':records,
        'exact_original_source_bytes_verified':True,'source_executed':False,
        'models_or_datasets_downloaded':False,'scope':manifest['scope']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=Path(__file__).parent/'references/verification_sources_manifest_v1.json')
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args();result=acquire(args.manifest,args.destination)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        with args.receipt.open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print('Exact pinned verification sources verified:',len(result['files']),
          '; newly downloaded:',sum(r['downloaded_this_invocation'] for r in result['files']),flush=True)


if __name__=='__main__':main()
