"""Article tables from COMPLETE frozen outputs, without new scientific analysis.

The declaration-only plan fixes every absolute family/endpoint cell and reuses
the six existing presentation matrices. Rendering exports every member value,
checks all original contrast estimates, and emits exact-ID TeX numeric macros.
No model ranking, changed contrast, new interval, inference or fitting occurs.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics

ROOT = Path(__file__).absolute().parent
BASE = ROOT/'artifacts/publication_20261017'
FINAL = BASE/'final_confirmation_v1'
REGISTRY_SHA = 'eb28c73d11482ad0881600a1e7d3fedee826f66f09b2e96e98f7de222881eb3f'
ATLAS = BASE/'final_contrast_atlas_v1/registry_eb28c73d1148_renderer_360ff1d96556/presentation_plan.json'
ATLAS_SHA = '8657b7902272dde724581d5b17e35ceefa5f9ee86a1bed4cfb86f4b5e9999574'
IMAGE = {'AUROC':'auc','average_precision':'average_precision',
    'recall_at_source_threshold':'recall','review_rate_at_source_threshold':'review_rate'}
LOCAL = {'positive_image_Dice','positive_image_IoU','normal_structure_pixel_FPR',
    'pooled_native_pixel_AP','box_point_hit_rate'}
SHORT = {'AUROC':'AUROC','average_precision':'Image AP',
    'recall_at_source_threshold':'Recall','review_rate_at_source_threshold':'Review',
    'false_positive_rate_at_source_threshold':'Image FPR','positive_image_Dice':'Pos. Dice',
    'positive_image_IoU':'Pos. IoU','normal_structure_pixel_FPR':'Structure FPR',
    'pooled_native_pixel_AP':'Native AP','known_crack_recall':'Known recall',
    'author_normal_FPR':'Normal FPR','box_point_hit_rate':'Box-point hit'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ref(p):
    return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))


def read(p):
    return json.loads(p.read_text())


def save(p,d):
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def table(p,rows):
    assert rows
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def close(a,b):
    assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)


def prepare():
    assert sha(FINAL/'frozen_registry.json')==REGISTRY_SHA and sha(ATLAS)==ATLAS_SHA
    registry=read(FINAL/'frozen_registry.json');atlas=read(ATLAS)
    declarations=registry['analysis_plan']['primary_comparisons']
    cells=list(dict.fromkeys((d[side],d['cohort'],d['analysis_stratum'],d['endpoint'])
        for d in declarations for side in ['first_family','second_family']))
    assert len(cells)==3686 and len(declarations)==6212
    core=[c for group in atlas['core_dense_question_matrices'] for c in group['contrast_ids']]
    assert len(core)==len(set(core))==456
    planned=dict(version=1,registry=ref(FINAL/'frozen_registry.json'),existing_atlas_plan=ref(ATLAS),
        reporter=ref(Path(__file__).absolute()),absolute_family_endpoint_cells=[
            dict(family=f,cohort=c,stratum=s,endpoint=e) for f,c,s,e in cells],
        all3686cells_retained=True,core_contrast_ids=core,core_tables=6,
        all6212_original_contrasts_receive_TeX_macros=True,ordering='Frozen declaration order; no outcome sorting.',
        display='Raw ratio units, four significant figures; scientific notation preserves small nonzero values. CSVs retain original float values.',
        means='Descriptive arithmetic mean of original per-fit endpoint values; never pooled logits or a new ensemble.',
        nulls='Any unsupported fitted-member endpoint makes that family mean not estimable; no imputation. Missing intervals retain original reasons.',
        no_new_fitting_inference_calibration_resampling_contrast_or_selection=True,
        target_values_read_in_prepare=False)
    out=BASE/'complete_confirmation_article_tables_v1'/sha(Path(__file__))[:20];out.mkdir(parents=True,exist_ok=True)
    path=out/'presentation_plan.json'
    if path.exists():assert read(path)==planned
    else:save(path,planned)
    copy=out/Path(__file__).name
    if not copy.exists():shutil.copy2(__file__,copy)
    assert sha(copy)==sha(Path(__file__))
    return registry,atlas,planned,out


def value(job,endpoint):
    if endpoint in IMAGE:return job['metrics'][IMAGE[endpoint]]
    if endpoint=='false_positive_rate_at_source_threshold':
        denominator=job['metrics']['fp']+job['metrics']['tn']
        return job['metrics']['fp']/denominator if denominator else None
    if endpoint in LOCAL:return job['localization_point_estimates'].get(endpoint)
    if endpoint in ['known_crack_recall','author_normal_FPR']:
        restricted=job['primary_restricted_endpoints']
        return restricted.get(endpoint) if restricted else None
    raise ValueError('Unknown frozen endpoint: '+endpoint)


def tex_number(number,signed=False):
    if number is None:return r'\mathrm{NE}'
    assert math.isfinite(number)
    text=format(number,'+.4g' if signed and number!=0 else '.4g')
    if 'e' in text:
        mantissa,exponent=text.split('e');return mantissa+r'\times10^{'+str(int(exponent))+'}'
    return text


def family_code(atlas,family):
    record=atlas['family_codes'][family]
    return record['code']


def cohort_title(spec):
    if spec['cohort'].startswith('scdd'):
        sizes={'whole':117,'ARTS':43,'CFVS':27,'CSIR':47}
        return 'SCDD '+spec['stratum']+' primary stratum, '+str(sizes[spec['stratum']])+' images'
    if spec['cohort'].startswith('ucf'):return 'UCF primary cohort, 507 images'
    return 'PVEL primary known-crack/author-normal cohort, 12,292 images'


def tex_escape(text):
    special={'\\':r'\textbackslash{}','_':r'\_\allowbreak{}','%':r'\%','&':r'\&',
             '#':r'\#','$':r'\$','{':r'\{','}':r'\}'}
    return ''.join(special.get(c,c) for c in text)


def render(registry,atlas,plan,out,review_path):
    review=read(review_path)
    assert review['final_analysis_checked'] and review['all6212_declarations_retained']
    analysis_path=FINAL/'primary_analysis.json'
    bindings=read(review_path.parent/'input_bindings.json')
    assert bindings[str(analysis_path.relative_to(ROOT))]==sha(analysis_path)
    analysis=read(analysis_path)
    assert analysis['inputs']['registry_id']==registry['registry_id']
    assert [r['declaration'] for r in analysis['primary_comparisons']]==registry['analysis_plan']['primary_comparisons']
    assert len(analysis['jobs'])==1683
    models={m['id']:m for m in registry['models']}
    families=defaultdict(list)
    for m in registry['models']:families[m['family']].append(m)
    jobs={(j['model_id'],j['cohort'],j['analysis_stratum']):j for j in analysis['jobs']}
    assert len(jobs)==1683
    run=out/('render_'+datetime.now(timezone.utc).isoformat().replace(':','-'));run.mkdir()
    save(run/'registration.json',dict(at=datetime.now(timezone.utc).isoformat(),
        plan=ref(out/'presentation_plan.json'),complete_original_analysis=ref(analysis_path),complete_review=ref(review_path)))
    absolute=[];members=[];by_cell={}
    for cell in plan['absolute_family_endpoint_cells']:
        f,c,s,e=[cell[k] for k in ['family','cohort','stratum','endpoint']]
        group=sorted(families[f],key=lambda m:m['seed']);per_seed=[]
        for model in group:
            job=jobs[model['id'],c,s];v=value(job,e)
            assert v is None or isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1
            members.append(dict(family=f,family_code=family_code(atlas,f),cohort=c,stratum=s,endpoint=e,
                model_id=model['id'],seed=model['seed'],status='estimated' if v is not None else 'not_estimable',
                value=v,analysis_images=job['primary_population']['analyzed_images']))
            per_seed.append(v)
        mean=statistics.mean(per_seed) if all(v is not None for v in per_seed) else None
        row=dict(family=f,family_code=family_code(atlas,f),cohort=c,stratum=s,endpoint=e,
            status='estimated' if mean is not None else 'not_estimable',unique_fitted_predictors=len(group),
            descriptive_fitted_mean=mean,seed11=per_seed[0],seed22=per_seed[1] if len(group)>1 else None,
            seed33=per_seed[2] if len(group)>1 else None,
            scope='Original matching primary population; per-fit metric means, no pooled model scores or new training replication.')
        absolute.append(row);by_cell[f,c,s,e]=row
    table(run/'all3686_absolute_family_endpoint_cells.csv',absolute)
    table(run/'all_original_fitted_member_values.csv',members)
    pair_values={(m['family'],m['cohort'],m['stratum'],m['endpoint'],m['model_id']):m['value'] for m in members}
    original_by_id={row['declaration']['id']:row for row in analysis['primary_comparisons']}
    checked_pairs=0
    for row in analysis['primary_comparisons']:
        d=row['declaration'];first=by_cell[d['first_family'],d['cohort'],d['analysis_stratum'],d['endpoint']]
        second=by_cell[d['second_family'],d['cohort'],d['analysis_stratum'],d['endpoint']]
        if row['status']=='estimated':
            assert first['status']==second['status']=='estimated'
            close(first['descriptive_fitted_mean'],row['estimate']['first_mean'])
            close(second['descriptive_fitted_mean'],row['estimate']['second_mean'])
            for pair in row['estimate']['fitted_pairs']:
                for side in ['first','second']:
                    close(pair_values[d[side+'_family'],d['cohort'],d['analysis_stratum'],d['endpoint'],pair[side+'_model_id']],pair[side+'_value'])
                checked_pairs+=1
        else:
            assert row['status']=='not_estimable' and 'not_estimable' in [first['status'],second['status']]
    macros=['% Complete original contrast values. Generated only after the full reporting gate.\n',
        '% Names use original contrast IDs; numbers remain raw ratios, not percentage points.\n']
    for row in analysis['primary_comparisons']:
        d=row['declaration'];est=row['estimate'];prefix='cf'+d['id'].removeprefix('contrast_')
        values={'First':tex_number(est['first_mean']) if est else r'\mathrm{NE}',
            'Second':tex_number(est['second_mean']) if est else r'\mathrm{NE}',
            'Delta':tex_number(est['mean_pair_difference'],True) if est else r'\mathrm{NE}'}
        ci=est['acquisition_family_percentile_ci95'] if est else None
        values['Lower']=tex_number(ci[0],True) if ci else r'\mathrm{NA}'
        values['Upper']=tex_number(ci[1],True) if ci else r'\mathrm{NA}'
        for suffix,number in values.items():
            macros.append(r'\expandafter\def\csname '+prefix+suffix+r'\endcsname{'+number+'}\n')
    (run/'all6212_original_contrast_number_macros.tex').write_text(''.join(macros))
    legend=[]
    for family in dict.fromkeys(cell['family'] for cell in plan['absolute_family_endpoint_cells']):
        record=atlas['family_codes'][family]
        legend.append(dict(family=family,family_code=record['code'],label=record['label'],
            unique_fitted_predictors=len(families[family]),seed_order=','.join(str(m['seed']) for m in sorted(families[family],key=lambda m:m['seed']))))
    table(run/'all427_family_code_legend.csv',legend);assert len(legend)==427
    fragments=[];core_rows=[]
    for number,spec in enumerate(atlas['core_dense_question_matrices'],1):
        text=[r'\begingroup\small\setlength{\tabcolsep}{3pt}'+'\n',r'\begin{longtable}{llrrrrl}'+'\n',
            r'\caption{'+cohort_title(spec)+r'. Complete fixed dense-control scope '+str(number)+r'. Raw endpoint ratios; differences are SECOND minus FIRST. Intervals are conditional and unadjusted. Codes resolve in the appended recipe legend.}\\'+'\n',
            r'\toprule FIRST $\to$ SECOND & Endpoint & FIRST & SECOND & Difference & Interval & Pair signs\\\midrule\endfirsthead'+'\n',
            r'\toprule FIRST $\to$ SECOND & Endpoint & FIRST & SECOND & Difference & Interval & Pair signs\\\midrule\endhead'+'\n']
        for identifier in spec['contrast_ids']:
            row=original_by_id[identifier];d=row['declaration'];est=row['estimate'];ci=est['acquisition_family_percentile_ci95'] if est else None
            a=family_code(atlas,d['first_family']);z=family_code(atlas,d['second_family'])
            interval=(r'\(['+tex_number(ci[0],True)+','+tex_number(ci[1],True)+r']\)') if ci else ('NE' if not est else 'point only' if d['cohort'].startswith('pvelad') or d['endpoint']=='pooled_native_pixel_AP' else 'unavailable')
            signs=','.join('+' if p['difference']>0 else '-' if p['difference']<0 else '0' for p in est['fitted_pairs']) if est else 'NE'
            first=tex_number(est['first_mean']) if est else r'\mathrm{NE}'
            second=tex_number(est['second_mean']) if est else r'\mathrm{NE}'
            delta=tex_number(est['mean_pair_difference'],True) if est else r'\mathrm{NE}'
            text.append(a+r'$\to$'+z+' & '+SHORT[d['endpoint']]+' & '+r'\('+first+r'\)'+' & '+r'\('+second+r'\)'+' & '+r'\('+delta+r'\)'+' & '+interval+' & '+signs+r'\\'+'\n')
            core_rows.append(dict(table=number,contrast_id=identifier,cohort=d['cohort'],stratum=d['analysis_stratum'],
                first_code=a,second_code=z,endpoint=d['endpoint'],status=row['status'],
                first=est['first_mean'] if est else None,second=est['second_mean'] if est else None,
                difference=est['mean_pair_difference'] if est else None,ci95_low=ci[0] if ci else None,
                ci95_high=ci[1] if ci else None,original_interval_status=est['interval_status'] if est else row['reason'],pair_signs=signs))
        text.append(r'\bottomrule\end{longtable}\endgroup'+'\n')
        fragment=run/f'core_table_{number:02d}.tex';fragment.write_text(''.join(text));fragments.append(fragment)
    assert len(core_rows)==456
    table(run/'all456_existing_core_matrix_comparisons.csv',core_rows)
    core_families=list(dict.fromkeys(f for spec in atlas['core_dense_question_matrices'] for pair in spec['family_pairs'] for f in pair))
    recipe_text=[r'\clearpage\section*{Recipe codes for every table above}'+'\n',
        r'\begin{longtable}{lp{0.84\linewidth}}\toprule Code & Original recipe description\\\midrule\endhead'+'\n']
    for family in core_families:
        recipe_text.append(family_code(atlas,family)+' & '+tex_escape(atlas['family_codes'][family]['label'])+r'\\'+'\n')
    recipe_text.append(r'\bottomrule\end{longtable}'+'\n')
    (run/'complete_core_tables.tex').write_text(r'\documentclass[10pt]{article}'+'\n'+
        r'\usepackage[margin=0.6in]{geometry}\usepackage{amsmath,booktabs,longtable}'+'\n'+
        r'\begin{document}\section*{Complete frozen dense-control numerical tables}'+'\n'+
        'All six pre-existing presentation matrix scopes; all 456 original comparisons, without outcome selection. Raw ratios are not percentages. Fitted-seed signs follow original pairs, including deterministic reuse; they are not additional training replication. No PVEL acquisition-family interval or pooled-native-AP interval is invented.\n'+
        '\n'.join(p.read_text() for p in fragments)+''.join(recipe_text)+r'\end{document}'+'\n')
    for b in registry['evaluator_files']:assert sha(ROOT/b['path'])==b['sha256']
    assert sha(analysis_path)==read(run/'registration.json')['complete_original_analysis']['sha256']
    save(run/'verification.json',dict(at=datetime.now(timezone.utc).isoformat(),registration=ref(run/'registration.json'),
        all_absolute_family_cells=len(absolute),fitted_member_endpoint_values=len(members),all6212_contrasts_crosschecked=True,
        original_fitted_pairs_crosschecked=checked_pairs,all6_core_tables_and456_original_comparisons=True,
        original_contrast_TeX_macros=6212*5,all427_family_codes_retained=True,core_family_codes_in_PDF=len(core_families),
        outputs=[ref(p) for p in sorted(run.iterdir()) if p.suffix in ['csv','tex']],
        new_model_inference_fitting_calibration_bootstrap_or_contrast=False,actual_PDF_compilation_and_visual_review_complete=False,
        main_manuscript_integrated=False,submission_ready=False))
    print(json.dumps(dict(output=str(run),absolute_family_cells=len(absolute),core_comparisons=456,original_contrast_macros=6212*5)))


def main():
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--render',action='store_true')
    p.add_argument('--verified-review',type=Path);args=p.parse_args()
    registry,atlas,plan,out=prepare()
    if args.prepare:
        print(json.dumps(dict(output=str(out),absolute_family_cells=3686,core_comparisons=456,target_values_read=False)))
    else:
        assert args.verified_review and args.verified_review.is_file()
        render(registry,atlas,plan,out,args.verified_review)


if __name__=='__main__':main()
