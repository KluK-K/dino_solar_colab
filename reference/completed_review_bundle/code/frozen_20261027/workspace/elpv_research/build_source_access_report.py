"""Author-source availability evidence and a local future-release profile."""
import html

from elresearch.common import ROOT, now, read_json, save_json, sha256

SOURCE=ROOT/'references/publication_20261017/source_access_review_20261018'
OUTPUT=ROOT/'artifacts/publication_20261017/source_access_and_reproduction_v1'


def main():
    source=read_json(SOURCE/'source_snapshots.json')
    for row in source['repositories']:
        if not row['recursive_tree_complete']:
            raise RuntimeError('Do not report a complete license-file inventory from a partial tree')
        for item in row['sources']:
            if 'sha256' in item and sha256(ROOT/item['path'])!=item['sha256']:
                raise RuntimeError('Original author-source snapshot changed')
            extracted=item.get('extracted_text')
            if extracted and sha256(ROOT/extracted['path'])!=extracted['sha256']:
                raise RuntimeError('Author request-form extraction changed')
    if source['model_license_bytes_equal'] is not True:
        raise RuntimeError('Review differing DINO license versions before stating equality')
    author_license=source['DINOv3_author_license']
    if sha256(ROOT/author_license['path'])!=author_license['sha256']:
        raise RuntimeError('Official DINO license snapshot changed')
    OUTPUT.mkdir(exist_ok=True)
    repos={r['name']:r for r in source['repositories']}
    descriptions={
        'ELPV':('CC BY-NC-SA4.0 images; Apache2.0 accompanying code, explicitly distinguished by the authors.',
            'Acquire images from the author endpoint. Preserve attribution/terms and original image checksums; no raw images or labels in a future code-only candidate.'),
        'SCDD':('No explicit license/terms filename found in the complete inspected imported tree. README requests dataset-paper citation. Article CC BY-NC-ND licensing is separate from linked raw-data terms.',
            'Use author acquisition recipes and versioned identities. Raw-image/annotation excerpts and redistribution permissions remain unresolved by this review; local research figures are not automatically a public-release grant.'),
        'UCF':('No explicit license/terms filename found in the complete inspected imported tree. README describes use of the data separately from the evaluation code.',
            'Point to the author release and pinned data identities; do not assign an unverified blanket license to externally obtained images, masks or labels.'),
        'PVEL_AD':('Repository LICENSE is Apache2.0. The separate author dataset request form restricts datasets to research/academic purposes and refers to a data agreement.',
            'Keep repository-code licensing separate from the externally hosted image archive. Record author access instructions and archive checksum; raw data are excluded from a code-only candidate.'),
        'PS_MT':('MIT license at the exact imported architecture revision; copyright2022 Yuyuan Liu.',
            'Retain the MIT license/copyright and identify local compatibility changes with any copied architecture code. Learned weights have separate source provenance.'),
        'PV_S3':('No explicit license/terms filename in the complete inspected pinned tree. Authors acknowledge the PS-MT basis.',
            'Keep upstream PV-S3 oracle files local. A future code-only candidate can supply pinned retrieval URLs/checksums for locally acquired verification sources, rather than bundling an unverified upstream license grant.')}
    rows=[]
    for name in ['ELPV','SCDD','UCF','PVEL_AD','PS_MT','PV_S3']:
        row=repos[name];terms,handling=descriptions[name]
        rows.append({'resource':name,'repository':row['repository'],'revision':row['revision_observed'],
            'revision_basis':row['revision_basis'],'observed_terms':terms,'future_candidate_handling':handling,
            'author_readme':next(s['url'] for s in row['sources'] if s['url'].endswith('/README.md')),
            'source_snapshot_paths':[s['path'] for s in row['sources'] if 'sha256' in s]})
    dino=source['DINOv3_pinned_HF_license_anonymous_request']
    if 'sha256' not in dino or sha256(ROOT/dino['path'])!=dino['sha256']:
        raise RuntimeError('Require the actual pinned-model license before recording its terms')
    rows.append({'resource':'DINOv3','repository':'https://github.com/facebookresearch/dinov3',
        'revision':'114c1379950215c8b35dfcd4e90a5c251dde0d32 (ViT-S/16)',
        'observed_terms':'Custom DINOv3 License. Research publication must acknowledge DINO use; covered-material redistribution must include the agreement and follow its terms.',
        'future_candidate_handling':'Cite Oriane Simeoni et al., DINOv3, arXiv2508.10104,2025; keep the exact license and record weights hashes. Obtain official weights through the recipient own authorized access; no credentials or pretrained weights in the code-only candidate.',
        'author_readme':'https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m',
        'source_snapshot_paths':[source['DINOv3_author_license']['path'],dino['path']]})
    profile={'at':now(),'kind':'Local future code-only release profile; nothing published',
        'author_source_review_sha256':sha256(SOURCE/'source_snapshots.json'),
        'allow_categories':['Research source under its eventual author-selected license',
            'Exact pinned dependency specification without copied dependency environments',
            'Acquisition recipes, split/group rules and identity checksums',
            'MIT PS-MT architecture notices with any copied architecture source',
            'DINOv3 license/acknowledgment and exact pretrained-weight identities',
            'Research tables/figures/outputs only after separate source and artifact review'],
        'exclude_path_patterns':['data/**/images/**','data/**/masks/**','data/**/pairs/**',
            '**/PV-S3_upstream_code/**','**/.venv/**','**/environment/**','**/source_capsules/**',
            '**/model.safetensors','**/pytorch_model.bin','**/token','**/stored_tokens'],
        'exclude_categories':['Raw source images, masks and unreviewed copied label collections',
            'Pretrained and learned model archives until their separate distribution terms are addressed',
            'User credentials, private account state and machine-local dependency environments',
            'Unverified-licensed PV-S3 upstream source; use author acquisition references for local oracle verification'],
        'external_verification_source_policy':'The full private offline capsule includes upstream numerical-oracle sources. A future code-only distribution must list original acquisition URLs and exact checksums and explicitly require local acquisition before running the same oracle checks; never skip those checks and call the smaller package equivalent.',
        'required_source_recipe_files':['fetch_verification_sources.py','references/verification_sources_manifest_v1.json','elresearch/licenses/PSMT_MIT_LICENSE.txt'],
        'explicit_local_setup':'python fetch_verification_sources.py --destination <project-root>',
        'pending':['All scientific source/final-confirmation results and complete paper package',
            'Author list, project-code licensing and final release artifact review',
            'Separate raw image/annotation excerpt and model-checkpoint distribution decisions'],
        'private_capsules_are_publication_ready':False,'published':False,
        'scope':'An inclusion specification, not an executed public package, license grant, permission request, external contact or legal determination. Scientific experiments and sealed confirmation protocol unchanged.'}
    save_json(OUTPUT/'future_code_only_profile_v1.json',profile)
    result={'at':now(),'builder_sha256':sha256(__file__),'author_source_snapshots_sha256':sha256(SOURCE/'source_snapshots.json'),
        'resources':rows,'DINOv3_official_and_pinned_license_bytes_match':source['model_license_bytes_equal'],
        'SCDD_article_license_evidence':{'text':'references/Pratt2023_SCDD.txt','sha256':sha256(ROOT/'references/Pratt2023_SCDD.txt'),
            'article_only_license':'CC BY-NC-ND; the paper points to the public author repository. This is not used to infer the raw external dataset license.'},
        'release_profile':str((OUTPUT/'future_code_only_profile_v1.json').relative_to(ROOT)),
        'research_images_or_weights_downloaded':False,'confirmation_scoring_performed':False,
        'scope':'Traceable author-source/access review. Absence of a repository license file does not prove absence of all permissions; repository-code terms are not silently extended to external image archives. Private/local figures are distinct from a public release.'}
    probe=OUTPUT/'source_reconstruction_verification.json'
    if probe.exists():
        proof=read_json(probe)
        if proof['fetcher_sha256']!=sha256(ROOT/'fetch_verification_sources.py') or proof['manifest_sha256']!=sha256(ROOT/'references/verification_sources_manifest_v1.json'):
            raise RuntimeError('Acquired source recipe changed after its real byte-verification probe')
        result['independent_source_reconstruction']={'proof':str(probe.relative_to(ROOT)),
            'sha256':sha256(probe),'four_fresh_author_downloads_exact':proof['four_fresh_author_downloads_exact'],
            'existing_exact_files_reused_without_network_download':proof['existing_exact_files_reused_without_network_download'],
            'intentional_mismatch_refused_and_unchanged':proof['intentional_mismatch_refused_and_unchanged'],
            'scope':proof['scope']}
    save_json(OUTPUT/'review.json',result)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['resource'],r['revision'],r['observed_terms'],r['future_candidate_handling']])+f'<td><a href="{html.escape(r["author_readme"])}">Author source</a></td></tr>' for r in rows)
    (OUTPUT/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Data, model and code source availability</title><style>body{font:15px/1.6 system-ui;color:#193e48;background:#f3f6f7;margin:0}main{max-width:1260px;margin:35px auto;padding:28px;background:white}table{border-collapse:collapse;font-size:12px}th,td{border-bottom:1px solid #dce5e7;padding:9px;text-align:left;vertical-align:top}a{color:#216d81}.scroll{overflow:auto}</style><main><h1>Data, models and third-party implementation: source and access evidence</h1><p>The paper reproduction package needs original source links, exact data/model identities and distinct terms for code, images and trained weights. This review snapshots complete author git trees, READMEs and identified license files. The original official and pinned DINOv3 license bytes match. All snapshot hashes are recorded; no user token, raw EL data, model weights or final predictions are accessed.</p><div class="scroll"><table><tr><th>Resource</th><th>Revision checked</th><th>Observed author terms</th><th>Future code-only handling</th><th>Primary evidence</th></tr>'''+table+'''</table></div><h2>Reproduction and public distribution are separate artifacts</h2><p>The existing private copied-source capsule passes its recorded implementation checks with locally available upstream oracle sources. It has not been published or declared a licensed public distribution. The future code-only profile excludes raw data, pretrained/learned weights, credentials and unverified-licensed upstream PV-S3 files. Exact author acquisition references can support local oracle verification before the same checks; omitting the oracle checks would not establish equivalent reproduction.</p><p>Article licensing is not silently assigned to external datasets. This review cannot establish every raw image-excerpt permission or an external dataset agreement. The final paper, authorship and actual release artifact remain unfinished. Current training, cohort membership, source/final decisions and score-sealing policy are unchanged.</p><p><a href="review.json">Resource evidence and local snapshot paths</a> · <a href="future_code_only_profile_v1.json">Local future inclusion profile</a> · <a href="https://arxiv.org/abs/2508.10104">DINOv3 technical report</a> · <a href="https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/blob/114c1379950215c8b35dfcd4e90a5c251dde0d32/LICENSE.md">Exact model-license revision</a></p></main></html>''')
    if probe.exists():
        page=OUTPUT/'report.html'
        detail='<h2>Exact local oracle-source reconstruction</h2><p>A concrete four-file author-source recipe is verified in a separate destination: the original PS-MT encoder/ResNet source and MIT notice, plus the reviewed PV-S3 numerical-loss oracle, reproduce the recorded SHA values. Repeated acquisition reuses exact files; an intentional mismatch is refused without overwriting. These probes do not execute upstream source, download models/data or run a new full-package test. <a href="source_reconstruction_verification.json">Actual byte-reconstruction and no-overwrite evidence</a>.</p>'
        page.write_text(page.read_text().replace('</main></html>',detail+'</main></html>'))
    print('Source/access review complete for7 resources; private/public artifact scope preserved',flush=True)
    return result


if __name__=='__main__':main()
