"""Finalize the reviewed article/code/aggregate delivery; no scientific computation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,importlib.util
ROOT=Path(__file__).absolute().parent
BASE=ROOT/'artifacts/publication_20261017'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))

def main():
    at=datetime.now(timezone.utc).isoformat()
    meta=read(ROOT/'manuscript/manuscript_v04_inputs.json');assert meta['revision']==61
    assert sha(ROOT/'manuscript/research_manuscript_v04.tex')==meta['manuscript_sha256']=='9276e86276c95a657f0acc426df75b6b7897a49abf62f8f6965cf341168114f6'
    pdf=ROOT/'output/pdf/article_v04_revision61_9276e86276c9'
    assert read(pdf/'actual_final_current_page_review.json')['all31_current_page_layouts_reviewed']
    replay=BASE/'saved_statistical_replay_v1/original_analysis_4d30aed026c4'
    assert read(replay/'verification.json')['state']=='complete_same_host_saved_output_analysis_repeatability'
    old=ROOT/read(BASE/'article_review_bundles_v2/latest.json')['folder']
    out=BASE/'completed_delivery_bundles_v3'/('revision61_'+at.replace(':','-'));out.mkdir(parents=True)
    exact=[]
    def capture(p,relative):
        q=out/relative;assert p.is_file() and not p.is_symlink() and not q.exists()
        q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
        exact.append(dict(path=str(q.relative_to(out)),sha256=sha(q),bytes=q.stat().st_size,source=ref(p)))
    changed={'ARTICLE_CONTEXT.json','README.txt','REPRODUCTION_REQUIREMENTS.json','verify_package.py',
        'article/research_manuscript_v04.tex','article/main_article.pdf','article/supplementary_record.pdf','article/research_manuscript_v04.pdf',
        'review/current31_page_layout_review.json','review/native_snapshot_compile_success.json'}
    for row in read(old/'MANIFEST.json')['files']:
        p=old/row['path'];assert sha(p)==row['sha256']
        if row['path'] not in changed:capture(p,row['path'])
    capture(ROOT/'manuscript/research_manuscript_v04.tex','article/research_manuscript_v04.tex')
    for n in ['main_article.pdf','supplementary_record.pdf','research_manuscript_v04.pdf']:capture(pdf/n,'article/'+n)
    capture(pdf/'actual_final_current_page_review.json','review/current31_page_layout_review.json')
    capture(pdf/'native_snapshot_compile_success.json','review/native_snapshot_compile_success.json')
    for n in ['raster_identity_comparison_to_revision60.json','supplement_identical_renderer_comparison.json']:capture(pdf/n,'review/'+n)
    capture(replay/'verification.json','review/completed_same_host_replay_verification.json')
    capture(replay/'replay_stdout.log','review/completed_same_host_replay_stdout.txt')
    capture(replay/'replay_stderr.log','review/completed_same_host_replay_stderr.txt')
    capture(Path(__file__).absolute(),'build_completed_delivery_bundle_v3.py')
    spec=importlib.util.spec_from_file_location('prior_builder',ROOT/'build_completed_article_review_bundle_v2.py')
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    spec1=importlib.util.spec_from_file_location('original_builder',ROOT/'build_article_review_bundle_v1.py')
    original=importlib.util.module_from_spec(spec1);spec1.loader.exec_module(original)
    intro='''COMPLETED SCIENTIFIC ARTICLE / PRIVATE REVIEW DELIVERY — REVISION61

The12-page English article and19-page current supplement integrate every frozen
confirmation result, all29 question themes and456 original dense-control-scope
contrasts. All31 page layouts have actual PDF QA: seven changed pages inspected
individually and24 unchanged pages reused only after exact RGB identity. The
byte-identical desktop compilation and standalone export passed. All nine
embedded numeric PDFs, historical narrative and original evidence categories
remain. The stale unfinished-status box is removed from the SAME owner source.

All1683 predictions, original full/saved-output audits and6212 declared contrasts
are complete. The ONE registered unchanged same-host saved-output replay PASSED
at2026-10-29T05:06:21Z; every job/contrast field matches apart from generation time
and original analysis bytes were not overwritten. It verifies same-host/environment/
implementation/saved-output repeatability, NOT independent implementation/device,
OS-isolation, fresh training or new inference reproduction. Do not launch another
queue, registry, replay or automation. The default recipient action is read/audit.

Scientific content, current pages and this local aggregate/code package are
complete for owner review. Formal authors, affiliations, correspondence and target
journal remain blank under the owner's explicit deferral. Journal-specific format,
declarations and final human author approval are therefore still to be finalized
before actual submission. No external contact/submission/release/payment occurred,
and no journal acceptance is promised. There is no scientific computation left
running in this completed workflow. Owner-only full input/output manifests stay
outside this distributable review package.

'''
    tail=prior.GUIDE[prior.GUIDE.index('COMPLETE NUMERIC EVIDENCE'):]
    tail=tail.replace('The ONE current\nsame-host replay uses the unchanged original analysis on saved outputs. Even if\nit passes, it establishes only same-host/environment/implementation repeatability;',
        'The ONE completed\nsame-host replay used the unchanged original analysis on saved outputs and passed.\nIt establishes only same-host/environment/implementation repeatability;')
    tail=tail.replace('The final authorized-input delivery evidence,\nsame-host replay completion and final delivery-state editorial pass remain pending.',
        'The owner-authorized saved-output statistical evidence and current article-page\nreview are complete. Full identities stay owner-only; author/journal finalization\nand an explicit future human submission decision remain separate.')
    assert 'remain pending' not in tail and 'Even if\nit passes' not in tail
    (out/'README.txt').write_text(intro+tail)
    req=read(old/'REPRODUCTION_REQUIREMENTS.json');req.update(at=at,version=3,
        same_host_saved_output_statistical_replay='complete_exact_match_all1683_jobs_and6212_contrasts_original_analysis_bytes_preserved',
        same_host_replay_verification=ref(replay/'verification.json'),
        independent_algorithm_cross_device_or_training_population_reproduction_complete=False)
    save(out/'REPRODUCTION_REQUIREMENTS.json',req)
    context=read(old/'ARTICLE_CONTEXT.json');context.update(at=at,manuscript_revision=61,source_sha256=meta['manuscript_sha256'],
        same_host_replay_completion_pending=False,authorized_input_statistical_replay_complete=True,
        scientific_content_and_current_page_review_complete=True,local_article_code_aggregate_delivery_complete=True,
        formal_author_and_venue_fields_deferred_by_owner=True,journal_specific_submission_ready=False,submission_ready=False)
    save(out/'ARTICLE_CONTEXT.json',context)
    checker=original.VERIFIER+prior.EXTRA_VERIFIER+'''
assert context['manuscript_revision']==61
assert context['authorized_input_statistical_replay_complete']
assert not context['same_host_replay_completion_pending']
assert context['scientific_content_and_current_page_review_complete']
replay=json.loads((root/'review/completed_same_host_replay_verification.json').read_text())
assert replay['returncode']==0 and replay['original_analysis_bytes_unchanged']
assert replay['all1683_jobs_and6212_contrasts_compared_by_unchanged_analyzer']
assert not context['journal_specific_submission_ready']
print(json.dumps(dict(scientific_content_and_same_host_statistical_replay_complete=True,
 current31_page_layout_review_complete=True,formal_author_journal_finalization_deferred=True,
 fresh_training_independent_algorithm_or_cross_device_reproduction=False,
 external_submission_performed=False,submission_ready=False)))
'''
    (out/'verify_package.py').write_text(checker)
    save(out/'EXACT_COPY_LEDGER.json',dict(at=at,files=exact))
    files=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]
    save(out/'MANIFEST.json',dict(at=at,files=files,scope='Completed scientific/current-article/code/aggregate owner review package. Full inputs excluded; formal author/journal finalization deferred.'))
    result=subprocess.run(['python3',str(out/'verify_package.py')],cwd=out,capture_output=True,text=True)
    (out/'verification_stdout.txt').write_text(result.stdout);(out/'verification_stderr.txt').write_text(result.stderr)
    save(out/'verification.json',dict(at=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,
        checks=[json.loads(x) for x in result.stdout.splitlines() if x.startswith('{')],
        scientific_workflow_rerun=False,external_upload=False,formal_author_journal_finalization_deferred=True))
    assert result.returncode==0,result.stderr
    assert 'still running' not in (out/'README.txt').read_text()
    save(out.parent/'latest.json',dict(at=at,folder=str(out.relative_to(ROOT)),manifest=ref(out/'MANIFEST.json'),verification=ref(out/'verification.json'),
        manuscript_revision=61,source_sha256=meta['manuscript_sha256'],files=len(files),
        scientific_content_and_same_host_statistical_delivery_complete=True,formal_author_journal_finalization_deferred=True,
        full_owner_input_output_identities_excluded=True,external_upload=False,submission_ready=False))
    print(json.dumps(dict(output=str(out),files=len(files),bytes=sum(x['bytes'] for x in files),checks=result.stdout),indent=2))

if __name__=='__main__':main()
