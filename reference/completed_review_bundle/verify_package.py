"""Standard-library package-byte/AST checks only. No research imports."""
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
assert r'\author{}' in (root/'article/research_manuscript_v04.tex').read_text()
context = json.loads((root/'ARTICLE_CONTEXT.json').read_text())
assert context['submission_ready'] is False and context['authorized_input_performance_reproduction_complete'] is False
assert context['full_registry_and_owner_input_output_inventories_included'] is False
print(json.dumps(dict(files_checked=checked, Python_sources_AST_checked=parsed,
    mandatory_four_author_file_recipe_preserved=True, mandatory_PV_S3_oracle_test_retained=True,
    raw_data_model_or_prediction_payloads_bundled=False, scientific_tests_or_research_performance_replay=False,
    scope='New bundle byte/AST/mandatory-recipe checks only; no implementation-suite, model or cross-device reproduction.',
    submission_ready=False)))

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
