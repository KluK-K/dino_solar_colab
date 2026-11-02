"""Saved-source training-accounting audit, without fitting, inference or final data."""
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from elresearch.common import ROOT, now, read_json, save_json, sha256, digest

BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'source_training_effort_v1'
FAMILIES = ['dinov3_frozen','dinov3_lora','dinov3_full','resnet18_full','resnet50_full','efficientnet_b0_full']
NAMES = ['DINOv3 frozen','DINOv3 LoRA','DINOv3 full','ResNet-18','ResNet-50','EfficientNet-B0']
SEEDS = [11,22,33]
BUDGETS = [.05,.1,.2,1.]


def saved_record(path, protocol, stage):
    run = read_json(path)
    if run['study_id'] != protocol['study_id']:
        raise RuntimeError('Run/study identity mismatch')
    history_path = path.parent/'history.json'
    history = read_json(history_path)
    count = run['completed_rounds']
    steps = protocol['updates_per_round']
    if count != len(history) or not 1 <= count <= protocol['maximum_rounds']:
        raise RuntimeError('Incomplete or inconsistent completed-round history')
    if run['optimizer_updates'] != count*steps:
        raise RuntimeError('Saved optimizer updates disagree with completed history')
    for i, row in enumerate(history,1):
        if row['round'] != i or row['optimizer_updates'] != steps*i:
            raise RuntimeError('Noncontiguous round/update history')
    if not 1 <= run['best_round'] <= count:
        raise RuntimeError('Selected round outside completed history')
    train, valid = run['training_ids'], run['validation_ids']
    if (len(set(train)) != len(train) or len(set(valid)) != len(valid)
            or set(train)&set(valid) or len(train) != run['training_images']
            or len(valid) != run['validation_images']):
        raise RuntimeError('Annotation identity/count mismatch')
    dense = stage == 'dense'
    trainable = (run['trainable_parameters'] if dense else
                 run['trainable_encoder_parameters']+run['head_parameters'])
    if not 0 < trainable <= run['total_parameters']:
        raise RuntimeError('Invalid fitted/total parameter counts')
    timer = 'training_and_validation_seconds_excluding_preparation' if dense else 'optimization_seconds'
    seconds = run[timer]
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError('Invalid recorded source timer')
    relative = path.relative_to(BASE/('dense_development_v1' if dense else stage))
    fold = 'ucf_calibration' if stage == 'calibration_v1' or relative.parts[0] == 'calibration' else relative.parts[1]
    presentations = run['optimizer_updates']*protocol['batch_size']
    row = dict(run_path=str(path.relative_to(ROOT)),stage=stage,fold=fold,
        family=run['model']['name'] if dense else run['model']['family'],
        configuration=run['model']['name'],seed=run['seed'],budget=run['training_budget_fraction'],
        training_images=len(train),validation_images=len(valid),annotation_kind='image+pixel' if dense else 'image',
        trainable_parameters=trainable,total_parameters=run['total_parameters'],
        best_round=run['best_round'],completed_rounds=count,
        optimizer_updates=run['optimizer_updates'],batch_size=protocol['batch_size'],
        sampled_training_presentations=presentations,
        average_presentations_per_original_training_image=presentations/len(train),
        recorded_seconds=seconds,timer_field=timer,
        cached_frozen_vectors=bool(run.get('cached_augmented_features',False)),
        preparation_and_frozen_encoding_in_timer=False,
        formal_end_to_end_training_cost=False)
    return row, [dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for p in [path,history_path]]


def figure(records, folder):
    selected = [r for r in records if r['stage']=='scdd_development_v1' and r['fold']=='internal']
    if len(selected)!=72:
        raise RuntimeError('Require all72 internal image fits; no favorable model/budget selection')
    for family in FAMILIES:
        for budget in BUDGETS:
            rows=[r for r in selected if r['family']==family and r['budget']==budget]
            if sorted(r['seed'] for r in rows)!=SEEDS:
                raise RuntimeError('Require each declared fitted seed')
    style={'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.edgecolor':'#9cabb0','axes.labelcolor':'#28444d',
        'xtick.color':'#3b5158','ytick.color':'#3b5158','pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,3,figsize=(14.1,5.3),gridspec_kw={'width_ratios':[1.03,1,1.05]})
        palette=['#267582','#ba8129','#765bb3','#b84f67']
        for i,family in enumerate(FAMILIES):
            rows=[r for r in selected if r['family']==family]
            trained={r['trainable_parameters'] for r in rows};total={r['total_parameters'] for r in rows}
            if len(trained)!=1 or len(total)!=1:raise RuntimeError('Parameter counts vary within one declared architecture')
            t,n=next(iter(trained)),next(iter(total))
            axes[0].plot([t,n],[i,i],color='#becbd0',lw=2,zorder=1)
            axes[0].scatter(n,i,s=62,marker='D',facecolors='white',edgecolors='#7c9199',zorder=2)
            axes[0].scatter(t,i,s=50,color='#246878',zorder=3)
            axes[0].annotate(f'{t:,}',(t,i),xytext=(0,9),textcoords='offset points',
                             fontsize=8.1,ha='center',color='#246878')
            for j,budget in enumerate(BUDGETS):
                group=sorted((r for r in rows if r['budget']==budget),key=lambda r:r['seed'])
                ys=[i+(-.27+j*.18)+(s-1)*.035 for s in range(3)]
                for ax,key in [(axes[1],'optimizer_updates'),(axes[2],'average_presentations_per_original_training_image')]:
                    ax.scatter([r[key] for r in group],ys,color=palette[j],s=27,
                               edgecolors='white',linewidths=.5,zorder=3)
        axes[0].set_xscale('log');axes[0].set_xlim(120,8e7)
        axes[0].set_xticks([1e3,1e5,1e7],['1K','100K','10M'])
        axes[0].set_xlabel('Parameters (log scale)')
        axes[0].set_yticks(range(6),NAMES)
        axes[0].scatter([],[],color='#246878',s=36,label='Fitted')
        axes[0].scatter([],[],facecolors='white',edgecolors='#7c9199',marker='D',s=40,label='Total')
        axes[0].legend(loc='upper left',bbox_to_anchor=(-.02,-.17),ncol=2,frameon=False,fontsize=9)
        axes[1].set_xlim(0,2100);axes[1].set_xticks([0,500,1000,1500,2000])
        axes[1].set_xlabel('Completed optimizer updates')
        axes[1].axvline(2000,color='#b6c1c5',lw=1,ls='--',zorder=1)
        axes[2].set_xscale('log');axes[2].set_xlim(.8,1600)
        axes[2].set_xticks([1,10,100,1000]);axes[2].xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        axes[2].set_xlabel('Training presentations / original image')
        for ax,title in zip(axes,['a  Fitted parameters and model size','b  Actual stopping work','c  Reuse of the training pool']):
            ax.set_title(title,loc='left',fontsize=11,pad=17)
            ax.set_ylim(5.55,-.65);ax.grid(axis='x',color='#e3e9eb',lw=.7);ax.set_axisbelow(True)
            if ax is not axes[0]:ax.set_yticks(range(6),[])
        for j,budget in enumerate(BUDGETS):
            axes[1].scatter([],[],color=palette[j],s=30,label=f'{100*budget:g}%')
        axes[1].legend(title='Training-image budget',loc='upper center',bbox_to_anchor=(1,-.17),
                       ncol=4,frameon=False,fontsize=9,title_fontsize=9)
        fig.subplots_adjust(left=.105,right=.985,top=.87,bottom=.35,wspace=.27)
        notes=[
            'All 72 internal image fits; three seed marks per budget. Completed updates include patience after the selected checkpoint. Counts are not FLOPs or controlled wall time.',
            'Presentations = updates × batch size (8); divided by original training-image count. Frozen presentations use cached D4 vectors, not new encoder forwards.',
            'Training budgets additionally use 120 validation labels and 574 upstream UCF calibration labels. Fitted parameters do not establish inference or memory savings.']
        for y,note in zip([.135,.092,.049],notes):fig.text(.105,y,note,fontsize=8.2,color='#536c75')
        for ext in ['png','pdf','svg']:
            fig.savefig(folder/f'internal_training_effort.{ext}',dpi=230,facecolor='white')
        plt.close(fig)


def main():
    if (BASE/'final_confirmation_v1/observation_opened.json').exists():
        raise PermissionError('Complete source accounting before independent observation')
    OUT.mkdir(exist_ok=True)
    plan_path=OUT/'plan.json'
    protocols={s:BASE/s/'protocol.json' for s in ['calibration_v1','scdd_development_v1','dense_development_v1']}
    plan=dict(kind='Retrospective source-metadata training effort audit; no model ranking or new inference',
        code_sha256=sha256(__file__),protocols={k:dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for k,p in protocols.items()},
        scientific_timer_sources=[dict(path=str((ROOT/'elresearch'/n).relative_to(ROOT)),sha256=sha256(ROOT/'elresearch'/n))
                                  for n in ['publication_train.py','publication_dense_train.py']],
        eligibility='All complete36 image calibration +180 image development +24 dense calibration; include dense development only through fully complete declared model/seed cells.',
        endpoints=['Fitted and total parameters','Completed and selected optimizer updates','Input presentations normalized by original labelled training images'],
        image_timer='After preparation/cache encoding; includes optimizer, per-round validation/checkpoint IO and final source prediction/calibration.',
        dense_timer='After source/bank preparation; includes optimizer/validation/checkpoint IO, native evaluation and fresh-model recovery/calibration.',
        timing_limit='Retain original seconds only as diagnostic logs, not a controlled end-to-end training benchmark. Preparation/cache/bank construction and parent adaptation need additional accounting; hardware/work differ.',
        presentation_limit='One sampled batch member is a training presentation; frozen classification uses a stored augmented feature. Mean per image is not a minimum per-image exposure or a physical full-dataset epoch.',
        future_scope='Keep this source-only accounting separate from the registered97-case end-to-end inference benchmark. Append complete dense cells without including partial outcomes.',
        final_confirmation_access=False)
    if plan_path.exists():
        if read_json(plan_path)!=plan:raise RuntimeError('Bound source-accounting plan changed')
    else:save_json(plan_path,plan)
    records=[];bindings=[]
    for stage,count in [('calibration_v1',36),('scdd_development_v1',180)]:
        state=read_json(BASE/stage/'status.json')
        paths=sorted((BASE/stage/'runs').rglob('run.json'))
        if state['state']!='complete' or len(paths)!=count:
            raise RuntimeError('Require whole completed image study')
        for p in paths:
            row,b=saved_record(p,read_json(protocols[stage]),stage);records.append(row);bindings+=b
    dense=read_json(protocols['dense_development_v1'])
    paths=sorted((BASE/'dense_development_v1/calibration').glob('*/run.json'))
    if len(paths)!=24:raise RuntimeError('Require all24 dense calibration fits')
    for p in paths:
        row,b=saved_record(p,dense,'dense');records.append(row);bindings+=b
    cells=read_json(BASE/'dense_development_complete_cells.json')
    for cell in cells['complete_cells']:
        if sha256(ROOT/cell['cell_path'])!=cell['cell_sha256']:
            raise RuntimeError('Complete native source cell changed')
        for binding in cell['identity']['bindings']:
            p=ROOT/binding['run']
            if sha256(p)!=binding['sha256']:raise RuntimeError('Bound source run changed')
            row,b=saved_record(p,dense,'dense');records.append(row);bindings+=b
    if len({r['run_path'] for r in records})!=len(records):raise RuntimeError('Repeated run identity')
    for p in protocols.values():bindings.append(dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)))
    bindings=sorted(bindings,key=lambda r:r['path'])
    identity=digest(dict(plan_sha256=sha256(plan_path),bindings=bindings))[:20]
    folder=OUT/identity;result_path=folder/'results.json'
    if result_path.exists():
        result=read_json(result_path)
        if result['input_bindings']!=bindings:raise RuntimeError('Existing source-accounting snapshot mismatch')
    else:
        folder.mkdir(exist_ok=True)
        with (folder/'source_fit_accounting.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        figure(records,folder)
        internal=[r for r in records if r['stage']=='scdd_development_v1' and r['fold']=='internal']
        full=next(r for r in internal if r['family']=='dinov3_full')
        lora=next(r for r in internal if r['family']=='dinov3_lora')
        summary=[]
        for family in FAMILIES:
            for budget in BUDGETS:
                group=[r for r in internal if r['family']==family and r['budget']==budget]
                summary.append(dict(family=family,budget=budget,training_images=sorted({r['training_images'] for r in group}),
                    seeds=[r['seed'] for r in sorted(group,key=lambda r:r['seed'])],
                    update_mean=mean(r['optimizer_updates'] for r in group),update_range=[min(r['optimizer_updates'] for r in group),max(r['optimizer_updates'] for r in group)],
                    mean_training_presentations_per_original=mean(r['average_presentations_per_original_training_image'] for r in group)))
        result=dict(at=now(),identity=identity,plan_sha256=sha256(plan_path),input_bindings=bindings,
            source_fits=len(records),image_fits=216,dense_calibration_fits=24,
            complete_dense_development_fits=len(records)-240,complete_dense_cells=len(cells['complete_cells']),
            internal_image_fits=72,internal_summary=summary,
            image_LoRA_fitted_parameters=lora['trainable_parameters'],image_full_fitted_parameters=full['trainable_parameters'],
            image_full_to_LoRA_fitted_parameter_ratio=full['trainable_parameters']/lora['trainable_parameters'],
            figures=[dict(path=str((folder/f'internal_training_effort.{ext}').relative_to(ROOT)),sha256=sha256(folder/f'internal_training_effort.{ext}')) for ext in ['png','pdf','svg']],
            final_confirmation_access=False,scientific_worker_modified=False,
            limits=[plan['timing_limit'],plan['presentation_limit'],'No fitting/inference/raw pixels/weights or additional model-performance endpoint. No uncertainty interval or efficiency superiority is inferred from these counts.'])
        save_json(result_path,result)
    save_json(OUT/'latest.json',dict(at=now(),identity=identity,results=str(result_path.relative_to(ROOT)),sha256=sha256(result_path),
        source_fits=result['source_fits'],complete_dense_cells=result['complete_dense_cells']))
    esc=html.escape
    summary_rows=''.join('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in [
        r['family'],f"{r['budget']*100:g}%",','.join(map(str,r['training_images'])),
        f"{r['update_mean']:.0f} [{r['update_range'][0]}, {r['update_range'][1]}]",f"{r['mean_training_presentations_per_original']:.1f}"] )+'</tr>'
        for r in result['internal_summary'])
    (OUT/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Source training effort</title>
<style>body{font:16px/1.5 system-ui;color:#213e48;max-width:1300px;margin:32px auto;padding:0 24px}img{width:100%}table{border-collapse:collapse;width:100%}th,td{padding:9px 12px;border-bottom:1px solid #d4dfe2;text-align:left}th{background:#eaf2f4}.note{background:#f2f5f6;padding:16px;border-left:4px solid #267582}a{color:#267582}</style>
<h1>What was fitted, and how much source training was performed</h1>
<p class="note">Parameter counts and optimizer updates describe fitted work. They do not establish end-to-end speed, memory or inference efficiency. Frozen classification trains on cached vectors; its source encoding occurs outside the solver timer.</p>
<img src="'''+identity+'''/internal_training_effort.png" alt="All72 internal image fits: fitted/total parameters, completed optimizer updates and training presentations per original labelled image">
<p>Image LoRA fits '''+f"{result['image_LoRA_fitted_parameters']:,}"+''' parameters versus '''+f"{result['image_full_fitted_parameters']:,}"+''' for full DINOv3 image adaptation ('''+f"{result['image_full_to_LoRA_fitted_parameter_ratio']:.1f}"+''' times fewer fitted parameters). The pretrained encoder remains part of the deployed model. The decoder-task counts are different and remain in the complete CSV.</p>
<p>The audit retains all36 image calibration fits,180 image development fits,24 dense calibration fits and every dense development fit from fully complete declared cells: '''+str(result['source_fits'])+''' source fits currently. Partial dense cells are excluded. All histories, update totals, annotation identities and parameter counts are checked against saved source protocols.</p>
<p>Each fraction uses additional source-validation labels/masks and upstream UCF hyperparameter-selection labels. No measured prospective annotation-time saving is implied. A round is50 fixed updates in these stages, not one sweep over the original images. Patience rounds after checkpoint selection are real training work.</p>
<table><tr><th>Image model</th><th>Training budget</th><th>Original training images</th><th>Completed updates, mean [range]</th><th>Mean presentations/original</th></tr>'''+summary_rows+'''</table>
<h2>Timing and computational limits</h2><p>Original per-run seconds are preserved in CSV as diagnostic logs. Image timers exclude preparation and frozen encoding; dense timers additionally include native evaluation and checkpoint-recovery checks after preparation. Initial references, cache construction, parent adaptation and different hardware/teacher passes cannot be hidden within those numbers. The registered97-case complete inference benchmark remains separate and pending.</p>
<p><a href="'''+identity+'''/source_fit_accounting.csv">Every source-fit record</a> · <a href="'''+identity+'''/internal_training_effort.pdf">Vector PDF</a> · <a href="'''+identity+'''/results.json">Bound source evidence</a> · <a href="plan.json">Recorded retrospective audit scope</a></p></html>''')
    print('Source training effort:',result['source_fits'],'fits;',result['complete_dense_cells'],'complete native cells; no new training/inference')


if __name__=='__main__':main()
