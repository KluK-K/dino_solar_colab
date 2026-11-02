"""Append presentation-corrected books, retaining both previous books and code.

Uses identical declarations, numbers, figure order and CSV bytes. Corrected
atlas recipe labels and single-panel joint margins supersede old presentation.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,shutil,subprocess
from pypdf import PdfReader,PdfWriter

ROOT=Path(__file__).absolute().parent
BASE=ROOT/'artifacts/publication_20261017'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-book-folder',type=Path);args=parser.parse_args()
    old=BASE/'complete_confirmation_companions_v1/d93a5f28cf8cf57edcc5/assembled_2026-10-29T01-42-24.650567+00-00'
    original=BASE/'final_contrast_atlas_v1/registry_eb28c73d1148_renderer_360ff1d96556'
    atlas=BASE/'final_contrast_atlas_v1/registry_eb28c73d1148_renderer_4a81e03f7ae7'
    diag=BASE/'complete_confirmation_diagnostics_v1/registry_eb28c73d1148_renderer_f22a9bf9d7e3/render_2026-10-29T01-27-08.463772+00-00'
    oldjoint=BASE/'complete_confirmation_question_readout_v1/37c319bc682b74421c43/render_2026-10-29T01-27-18.298509+00-00'
    joint=BASE/'complete_confirmation_question_readout_v1/6f1631e132f7c9506675/render_2026-10-29T01-55-22.391638+00-00'
    a=read(original/'presentation_plan.json');b=read(atlas/'presentation_plan.json')
    assert [k for k in a if a[k]!=b[k]]==['renderer','family_codes']
    assert read(atlas/'render_status.json')['all_rendered']
    for name in ['all_question_endpoint_direction_and_null_counts.csv','every_original_joint_endpoint_pair.csv']:
        assert sha(joint/name)==sha(oldjoint/name)
    rows=list(csv.DictReader((atlas/'family_codes.csv').open()));assert len(rows)==427 and not any(' | None | ' in x['label'] for x in rows)
    primary=BASE/'final_confirmation_v1/primary_analysis.json';assert sha(primary)=='382f595f14a6be6213dd8b0c65b6df4e4edf7d40725ba00afafb48ab496326a9'
    at=datetime.now(timezone.utc).isoformat();out=BASE/'complete_confirmation_companions_v3'/at.replace(':','-');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    save(out/'registration.json',dict(at=at,previous_companion=ref(old/'verification.json'),
        original_analysis=ref(primary),new_atlas_plan=ref(atlas/'presentation_plan.json'),
        reason='Full original recipe labels replace None/shared architecture labels; unclipped PVEL joint panels replace clipped presentation. Numbers, declarations, points and order are unchanged.',
        previous_books_retained=True,scientific_choices_changed=False))
    inputs=[('all6212_registered_contrasts.pdf',sorted((atlas/'figures').glob('*_atlas.pdf'))),
        ('all6_dense_control_matrices.pdf',sorted((atlas/'figures').glob('*_matrix.pdf'))),
        ('all_calibration_native_population_and_joint_diagnostics.pdf',sorted(diag.glob('*.pdf'))+sorted(joint.glob('joint_*.pdf')))]
    assert [len(x) for _,x in inputs]==[366,6,8]
    books=[]
    for name,paths in inputs:
        if args.reuse_book_folder:
            original_book=args.reuse_book_folder/name;assert original_book.is_file()
            shutil.copy2(original_book,out/name);assert sha(original_book)==sha(out/name)
        else:
            writer=PdfWriter()
            for p in paths:
                reader=PdfReader(p);assert len(reader.pages)==1;writer.add_page(reader.pages[0])
            with (out/name).open('wb') as f:writer.write(f)
        new=PdfReader(out/name)
        for i,p in enumerate(paths):
            assert new.pages[i].get_contents().get_data()==PdfReader(p).pages[0].get_contents().get_data()
        books.append(dict(book=ref(out/name),pages=len(paths),original_pages=[ref(p) for p in paths],all_assembled_page_content_streams_exact=True))
    tabledir=out/'tables';tabledir.mkdir();ledger=[]
    for p in sorted((old/'tables').rglob('*')):
        if not p.is_file():continue
        dest=tabledir/p.relative_to(old/'tables');dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest);assert sha(p)==sha(dest);ledger.append(dict(previous=ref(p),current=ref(dest)))
    legend=BASE/'complete_core_recipe_legend_clarification_v1/2026-10-29T01-53-16.190302+00-00'
    shutil.copy2(legend/'all427_unambiguous_recipe_codes.csv',tabledir/'all427_unambiguous_recipe_codes.csv')
    interpretation=BASE/'complete_question_interpretation_v1/2026-10-29T02-27-55.151373+00-00'
    shutil.copy2(interpretation/'all_question_contrast_interpretation_trace.csv',tabledir/'all_question_contrast_interpretation_trace.csv')
    save(out/'exact_numeric_table_copy_ledger.json',ledger)
    shutil.copy2(old/'page_to_original_contrast_index.json',out/'page_to_original_contrast_index.json')
    save(out/'verification.json',dict(at=at,original_analysis=ref(primary),all6212_contrasts_retained_once=True,
        books=books,exact_numeric_table_files=len(ledger),all29_question_interpretation=ref(interpretation/'review.json'),
        old_book_and_reporters_retained=True,all_original_values_intervals_seed_markers_and_axes_unchanged=True,
        all427_family_labels_unambiguous=True,corrected_joint_panels_no_data_change=True,
        actual_new_books_visual_QA_pending=True,submission_ready=False,external_upload=False))
    print(json.dumps(dict(folder=str(out),books=[dict(name=n,pages=len(p)) for n,p in inputs])))

if __name__=='__main__':main()
