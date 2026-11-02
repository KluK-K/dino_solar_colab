"""Build the current aggregate/article/code review bundle; never bundle owner inputs."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess

ROOT = Path(__file__).absolute().parent
BASE = ROOT/'artifacts/publication_20261017'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def save(p,d): p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def binding(p): return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))


GUIDE = '''CURRENT COMPLETE-CONFIRMATION ARTICLE / PRIVATE REVIEW BUNDLE

Read article/main_article.pdf (12 pages) and supplementary_record.pdf (19 pages).
The editable TeX is an exact copy of the owner's SAME manuscript, revision60.
The current author/affiliation/email/journal block is blank at the owner's request.
All nine embedded figure PDFs are retained; a XeLaTeX build requires no external
VAT asset or shell escape. The older development narrative remains default-off
inside that source, not deleted. The latest31-page layout review is in review/.
Layout review does not establish journal acceptance or methodological superiority.

All1683 frozen prediction jobs, original full/saved-output audits and all6212
declared contrasts are complete. The article integrates the complete findings,
all29 original question themes and all456 fixed dense-control-scope comparisons.
One original-host saved-output statistical repeatability replay is still running
at this dated bundle. Do not duplicate it. It is not new training or scoring.
The current package is reviewable but its final delivery gate remains pending.

COMPLETE NUMERIC EVIDENCE
figures/confirmation contains366 forest pages,6 dense matrices and8 all-record
calibration/native/population/joint diagnostic pages. tables/ retains all6212
ordered contrasts,16140 fitted-pair records,3686 absolute cells,2480 native
subgroups, all427 family descriptions and the overlapping29-theme trace. There
is no winner selection or resampling in packaging. Scores and intervals are the
original saved outputs; differences are SECOND minus FIRST. The forest figures
retain fitted seed markers. All380 pages have overview review, all six matrices
and declared representative forest templates have detail review. This does not
claim that every fine label on366 forests has been individually read.

figures/source_aggregate retains all14 source/cost/separate-ELPV numeric figures;
figures/source_dense now actually contains all three advanced source sheets and
their complete numeric companions. A packaging-only suffix bug left that latter
folder empty in the historical revision54 bundle and omitted those three files
from the revision55 private update despite narrative claims. Their original
scientific outputs were always present locally. The old bundle, exporter bytes
and manifests remain unchanged; the explicit amendment in review/ documents
the repair. This does not alter any experimental result or scientific code.

LIMITS THAT TRAVEL WITH THE RESULTS
427 recipe families and1061 model/inference views are not427 new architectures
or1061 newly trained models. Teacher/ensemble views share training, and identical
deterministic predictors do not supply new training replication. Both UCF/SCDD
development influenced design. Matched annotation identities do not match HPO,
pretraining, objectives, optimizer work or unlabeled use. Conditional acquisition-
family intervals are unadjusted for6212 comparisons, not independent-domain or
training-population intervals. Native pooled pixel AP is a cohort point estimate.
PVEL has no verified capture-family CI or mask Dice; UCF normal-structure FPR is
unsupported and stays null. An empty native denominator stays null. High PVEL
known-crack recall coexists with severe author-normal false positives. The source
and target conclusions are mixed; fewer trainable parameters do not prove speed.

CODE AND DEFAULT ACTION
Run python3 verify_package.py for byte/AST/coverage/mandatory-recipe checks only.
It uses the standard library, imports no research model and performs no network,
fit, inference, bootstrap, acquisition or environment installation. It is not the
historical171 or separate164 implementation tests, nor a research-performance
replication. The immutable code/frozen_20261027 candidate and its older context
are preserved. Current versioned reporters are in code/reporting. Historical
context files do not override this complete-analysis state. The recipient defaults
to reading/auditing, never another registry, scorer, training queue or automation.

AUTHORIZED REPLAY AND THE FOUR MANDATORY AUTHOR FILES
Read REPRODUCTION_REQUIREMENTS.json. A separately authorized working tree needs
the exact original input identities, source/fitted/official initializer weights,
archive bases, normal memories, raw data/annotations, policies and saved outputs.
Those payloads and full owner input/output inventories are deliberately absent.
Keep workspace/elpv_research plus workspace/elpv_midterm. Original absolute path
guards and the literal .venv/bin/python entry point must not be bypassed. No random
weights, invented labels, re-fitted target thresholds, symlink substitutions or
silent exclusions can stand in for missing inputs. The existing registry is
observed; a later authorized replay is not a new blind test. No656-course-test
ELPV inference and no DINOv2 execution are authorized.

The exact FOUR author-file recipe is at code/frozen_20261027/workspace/
elpv_research/references/verification_sources_manifest_v1.json. It pins PV-S3
losses.py, PS-MT encoder_decoder.py, resnet.py and LICENSE.md to commits/hashes.
Keep fetch_verification_sources.py and the mandatory PV-S3 numerical oracle
test_confidence_loss_matches_released_five_class_math in test_published_methods.py.
Unknown-licensed PV-S3 source is excluded. A recipient must review terms and
explicitly acquire the exact files in a separate authorized worktree. Missing
oracle input is a required failure, never an omitted test or full-suite pass.
The original owner already acquired and verified them; packaging does not refetch.

The owner verified26259 input files and11142 saved job files, including1683 result/
prediction pairs,3780 fixed maps and630 retirement receipts. These are byte/count/
identity closure checks, not independent numerical reproduction. The ONE current
same-host replay uses the unchanged original analysis on saved outputs. Even if
it passes, it establishes only same-host/environment/implementation repeatability;
no new-device, OS-isolation, independent algorithm or training-population claim.

No raw pixels/masks/label collections, weights/optimizer/RNG, prediction arrays/maps,
full registry/prepared/source bindings, private input/output inventories, environments
or credentials are included. Aggregate numerical tables grant no rights to original
datasets or code. Retain MIT PS-MT notices and artifact-specific DINOv3/data terms.
No blanket project license, external submission/contact/release/invitation/payment
or journal acceptance is implied. The final authorized-input delivery evidence,
same-host replay completion and final delivery-state editorial pass remain pending.
'''


EXTRA_VERIFIER = '''
import csv
def rows(relative):
    with (root/relative).open() as handle: return list(csv.DictReader(handle))
declarations=rows('tables/original_review/all6212_declared_questions.csv')
results=rows('tables/original_review/all6212_original_contrast_results.csv')
assert len(declarations)==len(results)==6212
assert [x['contrast_id'] for x in declarations]==[x['contrast_id'] for x in results]
trace=rows('tables/all_question_contrast_interpretation_trace.csv')
assert len(trace)==6252 and {x['id'] for x in trace}=={x['contrast_id'] for x in results}
assert len({x['theme'] for x in trace})==29
assert len(rows('tables/all427_unambiguous_recipe_codes.csv'))==427
assert len(rows('tables/article_tables/all456_existing_core_matrix_comparisons.csv'))==456
assert len(rows('tables/original_review/all_original_paired_seed_results.csv'))==16140
assert len(rows('tables/diagnostics/all2480_native_failure_subgroups.csv'))==2480
assert len(list((root/'figures/source_dense').glob('*.pdf')))==3
assert len(list((root/'figures/source_dense').glob('*.csv')))==3
assert len(list((root/'figures/source_aggregate').glob('*.pdf')))==14
assert context['primary_analysis_integrated_in_paper'] is True
assert context['current_main_pages']==12 and context['current_supplement_pages']==19
print(json.dumps(dict(all6212_ordered_contrasts_and29_themes_present=True,
    all456_fixed_core_contrasts_present=True,three_advanced_dense_source_PDFs_and_three_numeric_CSVs_actually_present=True,
    full_owner_input_output_inventories_excluded=True,scientific_replay_executed=False)))
'''


def main():
    meta=read(ROOT/'manuscript/manuscript_v04_inputs.json')
    source=ROOT/'manuscript/research_manuscript_v04.tex'
    assert meta['revision']==60 and sha(source)==meta['manuscript_sha256']
    primary=BASE/'final_confirmation_v1/primary_analysis.json'
    assert sha(primary)=='382f595f14a6be6213dd8b0c65b6df4e4edf7d40725ba00afafb48ab496326a9'
    registry=read(BASE/'final_confirmation_v1/frozen_registry.json')
    for row in registry['evaluator_files']: assert sha(ROOT/row['path'])==row['sha256']
    at=datetime.now(timezone.utc).isoformat()
    out=BASE/'article_review_bundles_v2'/('revision60_'+at.replace(':','-'))
    out.mkdir(parents=True)
    def capture(p,relative):
        assert p.is_file() and not p.is_symlink(),str(p)
        q=out/relative; assert not q.exists(),str(q)
        q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
    old_spec=importlib.util.spec_from_file_location('old_bundle_builder',ROOT/'build_article_review_bundle_v1.py')
    old=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old)
    pdf=ROOT/'output/pdf/article_v04_revision60_836fb452b83d'
    capture(source,'article/research_manuscript_v04.tex')
    for name in ['main_article.pdf','supplementary_record.pdf','research_manuscript_v04.pdf']:
        capture(pdf/name,'article/'+name)
    for name in ['native_snapshot_compile_success.json','raster_identity_comparison_to_revision59.json']:
        capture(pdf/name,'review/'+name)
    capture(ROOT/meta['latest_full_draft_page_review']['path'],'review/current31_page_layout_review.json')
    capture(ROOT/'output/pdf/complete_source_figure_companion_73fc35d50318/complete_source_figure_companion.pdf','figures/complete_source_figure_companion.pdf')
    ordered=read(ROOT/'output/pdf/complete_source_figure_companion_73fc35d50318/registration.json')['ordered_native_figures']
    for row in ordered:
        p=ROOT/row['original']['path'];assert sha(p)==row['original']['sha256']
        for ext in ['.pdf','.png','.svg']:capture(p.with_suffix(ext),'figures/source_aggregate/'+p.with_suffix(ext).name)
    for folder,name in [('complete_efficiency_figures_v1/f7412f405a7723e43416','efficiency_all_plotted_points.json'),
        ('completed_source_supplement_figures_v2/55a74938d4c270a5539f','UCF_and_ELPV_all_plotted_points.json')]:
        capture(BASE/folder/'plotted_points.json','figures/source_aggregate/'+name)
    dense=BASE/'advanced_source_figures_v1/311a03eb92d76d86e0fc'
    for p in sorted(dense.iterdir()):
        if p.suffix in ['.pdf','.png','.svg','.csv']:capture(p,'figures/source_dense/'+p.name)
    design=BASE/'study_design_figure_v1/3c41c15afbe9764df061'
    for ext in ['pdf','png','svg']:capture(design/('study_design.'+ext),'figures/design/study_design.'+ext)
    companions=BASE/'complete_confirmation_companions_v3/2026-10-29T02-36-35.431617+00-00'
    for p in sorted(companions.glob('*.pdf')):capture(p,'figures/confirmation/'+p.name)
    for p in sorted((companions/'tables').rglob('*')):
        if p.is_file():capture(p,str(p.relative_to(companions)))
    capture(companions/'page_to_original_contrast_index.json','review/confirmation_companion_page_index.json')
    capture(BASE/'complete_companion_visual_QA_v1/2026-10-29T02-38-05.145252+00-00/actual_visual_review.json','review/complete_companion_visual_review.json')
    candidate=ROOT/read(BASE/'reproduction_package_preparation_v1/latest_frozen_code_candidate.json')['folder']
    for p in sorted(candidate.rglob('*')):
        if p.is_file():capture(p,'code/frozen_20261027/'+str(p.relative_to(candidate)))
    reporters=[row['path'] for row in read(BASE/'manual_complete_reporting_chain_v1/analysis77662_v2_reporting_jsonparse/registration.json')['reporters']]
    reporters+=['build_frozen_confirmation_contrast_atlas_v2_recipe_labels.py','summarize_frozen_confirmation_questions_v2_layout.py',
        'build_frozen_confirmation_article_tables_v1.py','review_complete_confirmation_questions_v1.py','assemble_complete_confirmation_companions_v3_layout.py',
        'apply_manuscript_revision_v1.py','export_current_article_v1.py']
    for name in dict.fromkeys(reporters):capture(ROOT/name,'code/reporting/'+Path(name).name)
    capture(Path(__file__).absolute(),'build_completed_article_review_bundle_v2.py')
    capture(ROOT/'build_article_review_bundle_v1.py','historical/build_article_review_bundle_v1.py')
    old_bundle=BASE/'article_review_bundles_v1/revision54_2026-10-29T00-21-04.907577+00-00'
    assert len(list((old_bundle/'figures/source_dense').glob('*')))==0
    save(out/'review/packaging_suffix_amendment.json',dict(at=at,
        original_builder_sha256=sha(ROOT/'build_article_review_bundle_v1.py'),
        prior_manifest=binding(old_bundle/'MANIFEST.json'),actual_prior_advanced_dense_folder_files=0,
        defect="Path.suffix values include a leading dot; the original selector compared them to dotless strings.",
        repair="Version2 explicitly accepts .pdf/.png/.svg/.csv and verifies all3PDF+3CSV files.",
        original_scientific_outputs_unchanged=True,prior_bundle_manifests_and_code_preserved=True,
        prior_private_article55_same_selector_omission=True))
    save(out/'REPRODUCTION_REQUIREMENTS.json',dict(at=at,version=2,
        read_audit_default=True,receipt_scope='Byte/AST/coverage only; no model/scientific test/replay is executed by the bundle verifier.',
        exact_owner_input_files_verified=26259,exact_saved_job_files_verified=11142,
        result_prediction_pairs=1683,fixed_maps=3780,original_retirement_receipts=630,
        input_and_output_inventories_included=False,
        four_author_files=dict(recipe='code/frozen_20261027/workspace/elpv_research/references/verification_sources_manifest_v1.json',
            mandatory_PV_S3_oracle='test_confidence_loss_matches_released_five_class_math',
            exact_files=read(ROOT/'references/verification_sources_manifest_v1.json')['files'],
            unknown_licensed_PV_S3_source_included=False,missing_oracle_must_fail_not_skip=True),
        required_authorized_worktree_inputs=['original observed registry and full preparation/source identities',
            'all authorized raw datasets, metadata and masks/boxes','official pretrained initializers and exact fitted checkpoints/archive bases',
            'normal-reference memories, original optimizer/RNG/history where required','all saved result/prediction pairs and fixed maps',
            'both frozen population/native-failure policies and exact input/output identity closure'],
        original_host='32GiB Apple M2 Max/MPS',literal_owner_interpreter=str(ROOT/'.venv/bin/python'),
        original_absolute_path_guards_must_not_be_bypassed=True,
        same_host_saved_output_statistical_replay='one running owner-host replay; completion receipt pending',
        independent_algorithm_cross_device_or_training_population_reproduction_complete=False,
        no_new_registry_scoring_training_thresholds_or_course_test_access=True))
    (out/'verify_package.py').write_text(old.VERIFIER+EXTRA_VERIFIER)
    (out/'README.txt').write_text(GUIDE)
    save(out/'ARTICLE_CONTEXT.json',dict(at=at,manuscript_revision=60,source_sha256=sha(source),
        prediction_jobs_completed=1683,original_execution_audit_complete=True,primary_analysis_complete=True,
        primary_analysis_integrated_in_paper=True,declared_primary_contrasts=6212,recipe_families=427,model_or_view_records=1061,
        current_main_pages=12,current_supplement_pages=19,current_total_pages_layout_reviewed=31,
        all29_question_themes_interpreted=True,all456_fixed_core_contrasts_included=True,companion_PDF_pages=380,
        formal_author_and_venue_fields_deferred_by_owner=True,
        full_registry_and_owner_input_output_inventories_included=False,
        authorized_input_performance_reproduction_complete=False,same_host_replay_completion_pending=True,
        cross_device_reproduction_complete=False,submission_ready=False))
    files=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]
    save(out/'MANIFEST.json',dict(at=at,files=files,scope='Complete-aggregate/current-article/code private review bundle; final replay/delivery gate pending.'))
    result=subprocess.run(['python3',str(out/'verify_package.py')],cwd=out,capture_output=True,text=True)
    (out/'verification_stdout.log').write_text(result.stdout);(out/'verification_stderr.log').write_text(result.stderr)
    save(out/'verification.json',dict(at=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,
        checks=[json.loads(x) for x in result.stdout.splitlines() if x.startswith('{')],external_upload=False,
        new_model_scientific_test_or_replay_execution=False,submission_ready=False))
    assert result.returncode==0,result.stderr
    save(out.parent/'latest.json',dict(at=at,folder=str(out.relative_to(ROOT)),manifest=binding(out/'MANIFEST.json'),
        verification=binding(out/'verification.json'),manuscript_revision=60,source_sha256=sha(source),
        files=len(files),all3_advanced_source_figures_actually_included=True,full_owner_inventories_excluded=True,
        external_upload=False,submission_ready=False))
    print(json.dumps(dict(output=str(out),files=len(files),bytes=sum(x['bytes'] for x in files),checks=result.stdout,external_upload=False),indent=2))


if __name__=='__main__':main()
