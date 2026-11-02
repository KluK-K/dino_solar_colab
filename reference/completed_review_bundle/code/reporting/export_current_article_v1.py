"""Export the SAME saved TeX; extract and rasterize its current main article.

Creates a byte-identical build snapshot for the desktop compiler, without
opening a replacement document or accessing UI caches/preview screenshots.
Actual visual review is a separate human/model action, never inferred here.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess

import pypdf

ROOT = Path(__file__).absolute().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ref(p):
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def save(p, d):
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n')


def main():
    source = ROOT / 'manuscript/research_manuscript_v04.tex'
    meta = json.loads((ROOT / 'manuscript/manuscript_v04_inputs.json').read_text())
    digest = sha(source)
    assert digest == meta['manuscript_sha256']
    revision = meta['revision']
    out = ROOT / f'output/pdf/article_v04_revision{revision}_{digest[:12]}'
    out.mkdir(parents=True, exist_ok=False)
    at = datetime.now(timezone.utc).isoformat()
    command = [shutil.which('latexmk'), '-xelatex', '-interaction=nonstopmode',
               '-halt-on-error', '-latexoption=-no-shell-escape',
               '-outdir=' + str(out), str(source)]
    assert command[0]
    registration = dict(at=at, same_editable_source=str(source.relative_to(ROOT)),
                        source_sha256=digest, revision=revision, command=command,
                        working_directory=str(out.relative_to(ROOT)),
                        standalone_source_export=True,
                        native_editor_preview_or_export_claim=False,
                        no_UI_cache_screen_or_AppleScript_access=True,
                        scientific_choices_changed=False)
    save(out / 'export_registration.json', registration)
    shutil.copy2(__file__, out / Path(__file__).name)
    with (out / 'latexmk_stdout.log').open('w') as stdout, (out / 'latexmk_stderr.log').open('w') as stderr:
        result = subprocess.run(command, cwd=out, stdout=stdout, stderr=stderr)
    save(out / 'export_compile_result.json', dict(registration, returncode=result.returncode,
         completed_at=datetime.now(timezone.utc).isoformat(),
         PDF_exists=(out / 'research_manuscript_v04.pdf').exists(), actual_page_QA_complete=False))
    assert result.returncode == 0 and sha(source) == digest
    log = (out / 'research_manuscript_v04.log').read_text(errors='replace')
    bad = [x for x in log.splitlines() if re.search(r'(undefined references|undefined citations|multiply defined|LaTeX Warning: (Citation|Reference))', x, re.I)]
    assert not bad, bad
    aux = (out / 'research_manuscript_v04.aux').read_text()
    match = re.search(r'\\newlabel\{article_main_end\}\{\{[^}]*\}\{(\d+)\}', aux)
    assert match
    nmain = int(match.group(1))
    full = out / 'research_manuscript_v04.pdf'
    reader = pypdf.PdfReader(full)
    assert 1 <= nmain < len(reader.pages)
    for name, pages in [('main_article.pdf', range(nmain)),
                        ('supplementary_record.pdf', range(nmain, len(reader.pages)))]:
        writer = pypdf.PdfWriter()
        for page in pages:
            writer.add_page(reader.pages[page])
        with (out / name).open('wb') as f:
            writer.write(f)
    rasters = out / 'main_page_rasters'
    rasters.mkdir()
    renderer = shutil.which('pdftoppm')
    assert renderer
    render_command = [renderer, '-r', '144', '-png', str(out / 'main_article.pdf'), str(rasters / 'page')]
    subprocess.run(render_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    images = sorted(rasters.glob('page-*.png'))
    assert len(images) == nmain
    stage = out / 'source_snapshot'
    stage.mkdir()
    snapshot = stage / source.name
    shutil.copy2(source, snapshot)
    assert sha(snapshot) == digest
    warnings = [x.strip() for x in log.splitlines() if 'Overfull' in x or 'Underfull' in x]
    save(out / 'export_and_raster_verification.json', {
        'at': datetime.now(timezone.utc).isoformat(), 'revision': revision,
        'editable_source': ref(source), 'identical_build_snapshot': ref(snapshot),
        'standalone_same_source_compiled_PDF': ref(full),
        'main_article': ref(out / 'main_article.pdf'),
        'supplement': ref(out / 'supplementary_record.pdf'),
        'total_pages': len(reader.pages), 'main_article_pages': nmain,
        'supplement_pages': len(reader.pages) - nmain,
        'cross_references_and_citations_resolved': True,
        'remaining_TeX_box_warnings': warnings, 'render_command': render_command,
        'main_page_rasters': [ref(p) for p in images],
        'actual_visual_QA_complete': False, 'native_snapshot_compile_pending': True,
        'native_editor_preview_or_PDF_export_claim': False,
        'submission_ready': False,
    })
    print(json.dumps({'revision': revision, 'output': str(out), 'main_pages': nmain,
                      'supplement_pages': len(reader.pages)-nmain, 'warnings': warnings}, indent=2))


if __name__ == '__main__':
    main()
