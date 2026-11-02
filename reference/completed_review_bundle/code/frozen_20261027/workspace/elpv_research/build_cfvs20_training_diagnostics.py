"""Render every completed CFVS20 training history; no model or image inputs.

This is a retrospective reporting specification, not a new training protocol or
prospective hypothesis. Original registered closures, histories and results stay
unchanged. Terminal training diagnostics are distinguished from selected weights.
"""
import csv
import html
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from elresearch.common import ROOT, digest, now, read_json, save_json, sha256

BASE = ROOT / 'artifacts/publication_20261017'
OUT = BASE / 'CFVS20_training_diagnostics_v1'
CELL = BASE / 'published_complete_development_readout_v1/9ccdec9c331eff26374f/results.json'
FAMILIES = ['deeplabv3plus_full', 'dinov3_dense_frozen', 'dinov3_dense_lora',
            'dinov3_dense_reference', 'published_deeplab_fsl/primary',
            'published_psmt_ssl/primary']
LABELS = ['Adapted CNN', 'DINO frozen decoder', 'DINO LoRA decoder',
          'DINO normal reference', 'Full local port: supervised', 'Full local port: SSL teacher 1']
COLORS = {11: '#0072B2', 22: '#D55E00', 33: '#009E73'}


def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def main():
    state = read_json(BASE / 'inference_efficiency_v1/status.json')['state']
    assert state.startswith('queued'), 'Do not overlap formal timing jobs'
    assert not (BASE / 'final_confirmation_v1/observation_opened.json').exists()
    assert sha256(CELL) == '14fa09eded3b7b087d6919c34c05959a190988180ee0868c9040352488971472'
    result = read_json(CELL)
    assert result['unique_fits'] == 18 and len(result['all_seed_records']) == 42
    selected_rows = [r for r in result['all_seed_records']
                     if r['population'] == 'source_validation' and r['system'] in FAMILIES]
    assert len(selected_rows) == 18
    bindings = {b['path']: b for b in result['input_bindings']}
    jobs, inputs, histories = [], [ref(Path(__file__).absolute()), ref(CELL)], []
    for family in FAMILIES:
        rows = sorted((r for r in selected_rows if r['system'] == family), key=lambda r: r['seed'])
        assert [r['seed'] for r in rows] == [11, 22, 33]
        for row in rows:
            path = ROOT / row['run']
            assert sha256(path) == bindings[row['run']]['sha256']
            run = read_json(path)
            hp = path.parent / 'history.json'
            history = read_json(hp)
            assert len(history) == run['completed_rounds']
            assert [r['round'] for r in history] == list(range(1, len(history)+1))
            published = '/' in family
            fields = {'loss': 'training_loss' if published else 'training_joint_loss',
                      'AP': 'primary_source_selection_pixel_AP' if published else 'validation_selection_pixel_AP',
                      'AUROC': 'primary_source_image_AUROC' if published else 'validation_image_AUROC'}
            best = max(history, key=lambda r: r[fields['AP']])
            assert best['round'] == run['best_round']
            assert history[-1]['optimizer_updates'] == run['optimizer_updates']
            assert all(r['optimizer_updates'] == r['round'] * history[0]['optimizer_updates'] for r in history)
            assert all(math.isfinite(r[k]) for r in history for k in fields.values())
            source = row
            held = next(r for r in result['all_seed_records'] if r['run'] == row['run']
                        and r['population'] == 'held_acquisition_development' and r['system'] == family)
            record = {'family': family, 'seed': row['seed'], 'run': ref(path), 'history': ref(hp),
                      'selected_round': best['round'], 'terminal_round': len(history),
                      'selected_optimizer_updates': best['optimizer_updates'],
                      'terminal_optimizer_updates': history[-1]['optimizer_updates'],
                      'first_loss': history[0][fields['loss']], 'terminal_loss': history[-1][fields['loss']],
                      'selected_source_selection_AP': best[fields['AP']],
                      'selected_source_native_AP': source['pooled_native_pixel_AP'],
                      'source_empty_positive_images': source['empty_positive_images'],
                      'held_empty_positive_images': held['empty_positive_images'],
                      'held_positive_Dice': held['positive_image_Dice'],
                      'selected_pseudo_confident_fraction': best.get('pseudo_confident_fraction') if 'ssl' in family else None,
                      'terminal_pseudo_confident_fraction': history[-1].get('pseudo_confident_fraction') if 'ssl' in family else None,
                      'raw_pause_contaminated_rounds': [4] if 'ssl' in family and row['seed'] == 33 else []}
            jobs.append((record, history, fields))
            inputs += [ref(path), ref(hp)]
            for h in history:
                histories.append({'family': family, 'seed': row['seed'], 'round': h['round'],
                                  'optimizer_updates': h['optimizer_updates'],
                                  'original_selected_round': h['round'] == best['round'],
                                  **{k: h[v] for k, v in fields.items()},
                                  'supervised_CE': h.get('supervised_CE'),
                                  'unsupervised_ConfCE': h.get('unsupervised_ConfCE'),
                                  'pseudo_confident_fraction': h.get('pseudo_confident_fraction'),
                                  'active_teacher': h.get('active_teacher'),
                                  'raw_reported_round_seconds': h['seconds'],
                                  'round_seconds_pause_contaminated': h['round'] == 4 and 'ssl' in family and row['seed'] == 33})
    inputs += [ref(ROOT / 'elresearch/published_train.py'), ref(ROOT / 'elresearch/published_semisup.py'),
               ref(ROOT / 'elresearch/publication_dense_train.py')]
    plan = {'reporting_specification': 'Retrospective saved-history diagnostic, after observing the complete cell. No prospective significance or causal-mechanism claim.',
            'input_bindings': inputs, 'families': FAMILIES, 'seeds': [11, 22, 33],
            'all_18_original_fits': True, 'all_rounds_including_post_selected': True,
            'ensemble': 'Same SSL fit/history; show teacher1 selection history once. Retain both final inference views in the bound complete-cell result.',
            'axes': 'Optimizer updates, never wall time. Source-selection AP is sampled/preprocessed, not full native pooled AP. Loss objectives and work per update differ.',
            'confidence': 'Fraction of valid unlabelled pixels whose maximum teacher class probability passes0.6; any canonical class. Not crack coverage or pseudo-label accuracy. Selected and terminal values differ.',
            'timing': 'Preserve original raw seconds in CSV. SSLseed33 round4 pause-contaminated; no active-compute or formal speed comparison.',
            'new_fit_inference_bootstrap_or_selection': False}
    OUT.mkdir(exist_ok=True)
    plan_path = OUT / 'reporting_specification.json'
    if plan_path.exists():
        assert read_json(plan_path) == plan
    else:
        save_json(plan_path, plan)
    folder = OUT / digest(plan)[:20]
    if (folder / 'outputs.json').exists():
        for b in read_json(folder / 'outputs.json')['files']:
            assert sha256(ROOT / b['path']) == b['sha256']
        print({'reused': str(folder.relative_to(ROOT))})
        return
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(6, 3, figsize=(13, 17), layout='constrained')
    ap_max = math.ceil(max(h[f['AP']] for _, history, f in jobs for h in history)*10)/10
    for i, (family, title) in enumerate(zip(FAMILIES, LABELS)):
        for job, history, fields in jobs:
            if job['family'] != family:
                continue
            x = [h['optimizer_updates'] for h in history]
            selected = next(h for h in history if h['round'] == job['selected_round'])
            for j, field in enumerate(['loss', 'AP', 'AUROC']):
                ax = axes[i, j]
                ax.plot(x, [h[fields[field]] for h in history], color=COLORS[job['seed']], lw=1.6, label=f"Seed {job['seed']}")
                ax.scatter(selected['optimizer_updates'], selected[fields[field]], s=40,
                           facecolors='white', edgecolors=COLORS[job['seed']], zorder=3)
                ax.grid(alpha=.2)
                if j == 1:
                    ax.set_ylim(0, ap_max)
                elif j == 2:
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylim(bottom=0)
                ax.set_xlabel('Optimizer updates')
        axes[i, 0].set_ylabel(title+'\nTraining loss (own objective)')
        axes[i, 1].set_ylabel('Source selection pixel AP')
        axes[i, 2].set_ylabel('Source image AUROC')
    for j, title in enumerate(['Training objective', 'Sampled source-validation selection', 'Source-validation image ranking']):
        axes[0, j].set_title(title)
    axes[0, 0].legend(loc='upper right', fontsize=9)
    fig.suptitle('CFVS 20%: all 18 completed fits and every saved round\n50 training masks + 64 validation masks; SSL adds 199 source-training images\nOpen circles: original selected checkpoint; loss magnitudes are not cross-model performance scores', fontsize=13)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(folder / ('all18_training_histories.'+suffix), dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout='constrained')
    ssl = [(a, b, c) for a, b, c in jobs if a['family'] == 'published_psmt_ssl/primary']
    for i, (job, history, fields) in enumerate(ssl):
        x = [h['optimizer_updates'] for h in history]
        best = next(h for h in history if h['round'] == job['selected_round'])
        axes[i, 0].plot(x, [h['supervised_CE'] for h in history], label='Supervised CE', color='#0072B2')
        axes[i, 0].plot(x, [h['unsupervised_ConfCE'] for h in history], label='Unweighted ConfCE', color='#D55E00')
        axes[i, 1].plot(x, [h['pseudo_confident_fraction'] for h in history], color=COLORS[job['seed']])
        axes[i, 1].scatter(best['optimizer_updates'], best['pseudo_confident_fraction'], facecolors='white', edgecolors=COLORS[job['seed']], s=50)
        axes[i, 1].set_ylim(0, 1.05)
        axes[i, 2].plot(x, [h[fields['AP']] for h in history], color=COLORS[job['seed']])
        axes[i, 2].scatter(best['optimizer_updates'], best[fields['AP']], facecolors='white', edgecolors=COLORS[job['seed']], s=50)
        axes[i, 2].set_ylim(bottom=0)
        axes[i, 0].set_ylabel(f"Seed {job['seed']}\nCE / ConfCE")
        axes[i, 1].set_ylabel('Any-class confidence coverage')
        axes[i, 2].set_ylabel('Source selection pixel AP')
        axes[i, 2].text(.04, .97, f"Selected round {job['selected_round']}\nHeld empty masks: 158/158\nHeld positive Dice: 0", transform=axes[i, 2].transAxes, va='top', fontsize=9)
        for ax in axes[i]:
            ax.set_xlabel('Optimizer updates')
            ax.grid(alpha=.2)
            ax.axvline(best['optimizer_updates'], color='black', lw=.8, ls=':')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Complete CFVS 20% SSL fits: confidence is not crack recovery\nCoverage accepts any class at maximum teacher probability >= 0.6; class mix/accuracy not logged\nSelected-checkpoint outcomes differ from terminal-round training diagnostics; no causal collapse test', fontsize=12)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(folder / ('all3_SSL_diagnostics.'+suffix), dpi=150)
    plt.close(fig)
    for name, rows in [('all_rounds.csv', histories), ('all18_fit_summaries.csv', [r for r, _, _ in jobs])]:
        with (folder / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    findings = [
        'Every one of18fits has lower terminal training objective than its first round; trajectories are not necessarily monotone, losses differ and this is not a training-set accuracy or convergence proof.',
        'All3SSLteacher1 selected checkpoints have empty masks on all64source and158held images, despite finite nonzero optimizer updates. Selected rounds24/15/2 differ from terminal rounds30/21/8.',
        'Terminal any-class pseudo-confidence coverage is above99.8% in all3SSLfits. This is not crack sensitivity, class balance or pseudo-label accuracy, and does not show that selected checkpoints had terminal coverage.',
        'No class-wise pseudo-label counts or accuracy are present in saved histories. A self-training background-collapse cause is therefore not identified by these plots.',
        'Source-selection AP is sampled/preprocessed and chooses checkpoints; original full native AP and held metrics remain separate. No threshold, checkpoint, score direction, stopping rule or final model choice changes.',
        'SSLseed33 raw round4 duration remains pause-contaminated. Figures use optimizer updates, not elapsed seconds. One update has different work across objectives, networks and supervision; no equal-compute or speed claim.']
    report = {'at': now(), 'specification': ref(plan_path), 'input_bindings': inputs,
              'unique_fits': 18, 'seeds': [11, 22, 33], 'saved_rounds': len(histories),
              'fit_summaries': [r for r, _, _ in jobs], 'findings': findings,
              'new_fit_inference_bootstrap_or_selection': False, 'independent_confirmation': False}
    save_json(folder / 'results.json', report)
    content = '<!doctype html><html lang="en"><meta charset="utf-8"><title>CFVS20 saved training histories</title><style>body{font:16px Arial;max-width:1300px;margin:30px auto;padding:10px}img{max-width:100%}</style><h1>All completed CFVS20 training histories</h1><p>Retrospective diagnostic; no model inference or independent confirmation. All18fits/all3seeds retained.</p>'
    content += ''.join('<p>'+html.escape(x)+'</p>' for x in findings)
    content += '<img src="all18_training_histories.png" alt="All six families and three seeds"><img src="all3_SSL_diagnostics.png" alt="All three SSL training diagnostics"></html>'
    (folder / 'report.html').write_text(content)
    files = [ref(p) for p in sorted(folder.iterdir()) if p.is_file()]
    save_json(folder / 'outputs.json', {'at': now(), 'files': files, 'visual_QA_pending': True})
    save_json(OUT / 'latest.json', {'at': now(), 'directory': str(folder.relative_to(ROOT)),
                                  'result': ref(folder / 'results.json'), 'outputs': ref(folder / 'outputs.json'),
                                  'visual_QA_pending': True, 'manuscript_integrated': False})
    print({'folder': str(folder.relative_to(ROOT)), 'fits': 18, 'saved_rounds': len(histories),
           'result': ref(folder / 'results.json'), 'visual_QA_pending': True})


if __name__ == '__main__':
    main()
