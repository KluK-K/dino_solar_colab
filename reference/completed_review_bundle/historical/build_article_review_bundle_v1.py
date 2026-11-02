"""Assemble the current paper/source figures and code-only review materials.

This is a local, incomplete article-review bundle. It excludes authorized raw
inputs, full scientific/input inventories and learned/prediction payloads. It
does not install, import research code, run tests/models, or claim reproduction.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).absolute().parent
BASE = ROOT/'artifacts/publication_20261017'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def save(p, d):
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


VERIFIER = '''"""Standard-library package-byte/AST checks only. No research imports."""
from pathlib import Path
import ast, hashlib, json

root = Path(__file__).absolute().parent
manifest = json.loads((root/'MANIFEST.json').read_text())
checked = parsed = 0
for row in manifest['files']:
    p = root/row['path']
    assert p.is_file() and not p.is_symlink(), row['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256'], row['path']
    assert p.stat().st_size == row['bytes'], row['path']
    if p.suffix == '.py':
        ast.parse(p.read_text(), filename=row['path']); parsed += 1
    checked += 1
oracle = root/'code/frozen_20261027/workspace/elpv_research/references/verification_sources_manifest_v1.json'
assert len(json.loads(oracle.read_text())['files']) == 4
tests = root/'code/frozen_20261027/workspace/elpv_research/tests/test_published_methods.py'
assert any(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name ==
    'test_confidence_loss_matches_released_five_class_math' for x in ast.walk(ast.parse(tests.read_text())))
assert r'\\author{}' in (root/'article/research_manuscript_v04.tex').read_text()
context = json.loads((root/'ARTICLE_CONTEXT.json').read_text())
assert context['submission_ready'] is False and context['authorized_input_performance_reproduction_complete'] is False
assert context['full_registry_and_owner_input_output_inventories_included'] is False
print(json.dumps(dict(files_checked=checked, Python_sources_AST_checked=parsed,
    mandatory_four_author_file_recipe_preserved=True, mandatory_PV_S3_oracle_test_retained=True,
    raw_data_model_or_prediction_payloads_bundled=False, scientific_tests_or_research_performance_replay=False,
    scope='New bundle byte/AST/mandatory-recipe checks only; no implementation-suite, model or cross-device reproduction.',
    submission_ready=False)))
'''

GUIDE = '''ARTICLE REVIEW BUNDLE — PRIVATE LOCAL DRAFT, REVISION 54

Read article/main_article.pdf first. It contains the concise English article;
author/affiliation/email/journal fields are intentionally blank under the owner's
instruction. The original editable manuscript remains in its research folder.
The TeX here is an exact archived copy of that same source, with all embedded
figure PDFs. A normal XeLaTeX build needs no external image asset or shell escape.

article/supplementary_record.pdf retains the clearly identified historical
development/protocol record and the complete numerical source-figure appendix.
Dated historical status statements are preserved history. ARTICLE_CONTEXT.json
states the current scope. The full source export is also retained. Every one of
its 61 current draft pages was actually inspected as a saved PDF rasterization.
This page review checks layout; it does not finish the scientific analysis.

CURRENT SCIENCE
All 1,683 frozen prediction jobs and their original full audit are complete.
The unchanged original all-6,212-contrast analysis is still running when this
draft is packaged. No partial target result is promoted to a final finding.
The remaining work includes complete declaration/seed/null/native-failure and
population review, quantitative confirmation figures, final interpretation and
integration in the SAME manuscript, and another final compile/export/page review.
This bundle is not a submission-ready paper or a public release.

FIGURES AND NUMERICAL COMPANIONS
figures/source_aggregate contains every one of the 14 original reviewed source,
formal-cost and separate ELPV aggregate figures in PDF/PNG/SVG, without ranking or
outcome selection. The native-size fourteen-page companion PDF permits zoom.
figures/source_dense contains all three advanced dense-source figures and their
complete fitted-seed, contrast and paired-seed numerical tables. The study-design
diagram is in figures/design. These are completed development/cost/CV results;
they do not constitute independent confirmation. The normal-reference and CNN
results remain mixed across tasks, acquisitions, budgets and fitted seeds.

READING THE RESULTS
Differences use SECOND family minus FIRST family. A positive difference does not
automatically mean better performance: review/FPR must be read with recall.
Intervals condition on fitted systems and are unadjusted across the many declared
comparisons. They do not describe training-population or independent-domain
uncertainty. Native pooled pixel AP remains a whole-cohort point estimate, not a
pixel bootstrap. PVEL has no verified acquisition-family interval, and its box
truth cannot support mask Dice. UCF normal-structure FPR remains null. Both UCF
and SCDD development informed study-wide design. Matching original mask IDs does
not match pretraining, search effort, objectives, optimizer work or unlabeled use.

CODE AND THE SAFE DEFAULT
The code/frozen_20261027 tree is the immutable earlier code-only candidate, with
its original manifest, context and source-acquisition recipe intact. Its older
FROZEN_CONTEXT.json predates prediction completion; it is not the current status.
The additional four reporting scripts are exact copies of the currently bound
reporters. No manual worker launcher, recurring automation, registry builder
instruction or active PID is supplied as a recipient action.

Run python3 verify_package.py from this bundle only for the standard-library
byte/AST/mandatory-recipe check. It imports no research code and performs no model
load, fit, inference, bootstrap, network access or environment installation.
These new packaging checks are not the historical 171 implementation checks or
the separate 164 fresh-environment checks. Those earlier checks were same-host
synthetic/metadata/oracle checks, not OS isolation, cross-device performance
reproduction or tests of every new reporter.

AUTHORIZED INPUTS AND MANDATORY ORACLES
Research-performance replay requires separately authorized original datasets,
metadata, the ONE observed registry, exact pretrained/fitted weights and archive
bases, memories, optimizer/RNG states, saved predictions, policies and identities.
They are deliberately absent. Do not fabricate inputs, reconstruct random bases,
create a second registry, refit thresholds, launch duplicate training/scoring or
infer new ELPV course-test outputs. The original host is the 32 GiB M2 Max/MPS;
cross-host compatibility and performance replay have not been demonstrated.

The exact four-author-file recipe and acquisition code remain in
code/frozen_20261027/workspace/elpv_research/references/
verification_sources_manifest_v1.json and fetch_verification_sources.py.
The mandatory PV-S3 oracle is retained in tests/test_published_methods.py as
test_confidence_loss_matches_released_five_class_math. Unknown-licensed upstream
PV-S3 source is excluded; an authorized user must review terms and acquire the
exact files separately. Missing oracle inputs must be an explicit failure, not
a skipped check or a passing full-suite claim. The owner already has verified
exact files; this packaging step does not download them again.

DISTRIBUTION AND LIMITS
Raw EL pixels, masks, label collections, full registry/prepared/source-input
inventories, owner-only saved-output inventories, learned/optimizer/RNG payloads,
prediction arrays/maps, credentials and environments are absent. Aggregate
numerical points, code and paper text do not grant rights to external inputs.
Retain the copied PS-MT MIT notice. DINOv3, data and other source terms remain
artifact-specific; this bundle assigns no blanket project/upstream/data license.
No external contact, public release, invitation, submission or paid compute is
performed or authorized by this package. The default recipient action is reading
and auditing. A final authorized-input package and final statistical closure are
still required; manuscript and scientific records remain append-only preserved.
'''


def main():
    meta = read(ROOT/'manuscript/manuscript_v04_inputs.json')
    source = ROOT/'manuscript/research_manuscript_v04.tex'
    assert meta['revision'] == 54 and sha(source) == meta['manuscript_sha256']
    frozen = BASE/'final_confirmation_v1/frozen_registry.json'
    assert sha(frozen) == 'eb28c73d11482ad0881600a1e7d3fedee826f66f09b2e96e98f7de222881eb3f'
    registry = read(frozen)
    for row in registry['evaluator_files']:
        assert sha(ROOT/row['path']) == row['sha256']
    code_pointer = read(BASE/'reproduction_package_preparation_v1/latest_frozen_code_candidate.json')
    code = ROOT/code_pointer['folder']
    assert sha(code/'CODE_MANIFEST.json') == code_pointer['manifest']['sha256']
    code_manifest = read(code/'CODE_MANIFEST.json')
    at = datetime.now(timezone.utc).isoformat()
    out = BASE/'article_review_bundles_v1'/('revision54_'+at.replace(':','-'))
    out.mkdir(parents=True)
    for child in ['article','figures/source_aggregate','figures/source_dense','figures/design','code/reporting']:
        (out/child).mkdir(parents=True)
    def capture(p, relative):
        assert p.is_file() and not p.is_symlink()
        q=out/relative; q.parent.mkdir(parents=True,exist_ok=True)
        assert not q.exists(); shutil.copy2(p,q); assert sha(q)==sha(p)
    capture(source,'article/research_manuscript_v04.tex')
    pdf = ROOT/'output/pdf/article_v04_revision54_a9ef20565fb7'
    for name in ['main_article.pdf','supplementary_record.pdf','research_manuscript_v04.pdf']:
        capture(pdf/name,'article/'+name)
    capture(ROOT/'output/pdf/complete_source_figure_companion_73fc35d50318/complete_source_figure_companion.pdf',
            'figures/complete_source_figure_companion.pdf')
    companion = read(ROOT/'output/pdf/complete_source_figure_companion_73fc35d50318/registration.json')
    for row in companion['ordered_native_figures']:
        original=ROOT/row['original']['path']; assert sha(original)==row['original']['sha256']
        for ext in ['pdf','png','svg']:
            p=original.with_suffix('.'+ext)
            capture(p,'figures/source_aggregate/'+p.name)
    for folder, name in [
        ('complete_efficiency_figures_v1/f7412f405a7723e43416','efficiency_all_plotted_points.json'),
        ('completed_source_supplement_figures_v2/55a74938d4c270a5539f','UCF_and_ELPV_all_plotted_points.json')]:
        capture(BASE/folder/'plotted_points.json','figures/source_aggregate/'+name)
    dense=BASE/'advanced_source_figures_v1/311a03eb92d76d86e0fc'
    for p in sorted(dense.iterdir()):
        if p.suffix in ['pdf','png','svg','csv']:
            capture(p,'figures/source_dense/'+p.name)
    design=BASE/'study_design_figure_v1/3c41c15afbe9764df061'
    for ext in ['pdf','png','svg']:
        capture(design/('study_design.'+ext),'figures/design/study_design.'+ext)
    # The existing candidate is an explicit, already reviewed code-only boundary.
    # Copy every file while preserving its exact original relative layout.
    for p in sorted(code.rglob('*')):
        if p.is_file():
            capture(p,'code/frozen_20261027/'+str(p.relative_to(code)))
    chain=BASE/'manual_complete_reporting_chain_v1/analysis77662_v2_reporting_jsonparse/registration.json'
    for row in read(chain)['reporters']:
        p=ROOT/row['path']; assert sha(p)==row['sha256']
        capture(p,'code/reporting/'+p.name)
    capture(Path(__file__).absolute(),'build_article_review_bundle_v1.py')
    (out/'verify_package.py').write_text(VERIFIER)
    (out/'README.txt').write_text(GUIDE)
    save(out/'ARTICLE_CONTEXT.json',dict(at=at,manuscript_revision=54,source_sha256=sha(source),
        prediction_jobs_completed=1683,prediction_jobs_planned=1683,original_execution_audit_complete=True,
        primary_analysis_integrated_in_paper=False,declared_primary_contrasts=6212,
        original_analysis_already_running_do_not_duplicate=True,recipe_families=427,model_or_view_records=1061,
        current_draft_total_PDF_pages_actually_viewed=61,formal_author_and_venue_fields_deferred_by_owner=True,
        original_code_candidate_is_historical=True,four_author_file_recipe_and_PV_S3_oracle_mandatory=True,
        full_registry_and_owner_input_output_inventories_included=False,
        authorized_input_performance_reproduction_complete=False,cross_device_reproduction_complete=False,
        scientific_superiority_or_acceptance_promised=False,submission_ready=False))
    files=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size)
           for p in sorted(out.rglob('*')) if p.is_file()]
    save(out/'MANIFEST.json',dict(at=at,files=files,scope='Current local article/code-only review bundle; not final authorized-input research reproduction or submission package.'))
    result=subprocess.run(['python3',str(out/'verify_package.py')],capture_output=True,text=True,cwd=out)
    (out/'verification_stdout.log').write_text(result.stdout)
    (out/'verification_stderr.log').write_text(result.stderr)
    save(out/'verification.json',dict(at=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,
        checks=json.loads(result.stdout) if result.returncode==0 else None,scientific_files_changed=False,
        model_or_environment_execution=False,external_upload=False,submission_ready=False))
    assert result.returncode==0,result.stderr
    for row in registry['evaluator_files']:
        assert sha(ROOT/row['path'])==row['sha256']
    print(json.dumps(dict(output=str(out),files=len(files),bytes=sum(x['bytes'] for x in files),
        new_bundle_checks=json.loads(result.stdout),external_upload=False),indent=2))


if __name__=='__main__':
    main()
