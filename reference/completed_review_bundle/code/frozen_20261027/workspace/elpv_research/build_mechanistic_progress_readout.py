"""Append complete source mechanism cells using saved results, never model inference."""
import csv
import html
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from elresearch.common import ROOT, read_json, save_json, sha256, digest, now
from elresearch.metrics import classification


BASE = ROOT / 'artifacts/publication_20261017'
SOURCE = BASE / 'mechanistic_controls_v1'
OUT = BASE / 'mechanistic_complete_cells_v1'
NEW = ['dinov3_patch_mlp', 'dinov3_global_normal_distance',
       'dinov3_postlora_no_reference', 'dinov3_postlora_reference']
NAMES = {'deeplabv3plus_full': 'Adapted CNN', 'dinov3_dense_frozen': 'Frozen decoder',
         'dinov3_dense_lora': 'LoRA decoder', 'dinov3_dense_full': 'Full adaptation',
         'dinov3_dense_reference': 'Frozen + reference',
         'dinov3_reference_no_position': 'Reference, no position',
         'dinov3_reference_mean_evidence': 'Reference, mean evidence',
         'dinov3_patch_mlp': 'Patch MLP', 'dinov3_global_normal_distance': 'Normal distance *',
         'dinov3_postlora_no_reference': 'Post-LoRA, fresh decoder',
         'dinov3_postlora_reference': 'Post-LoRA + reference'}
METRICS = [('AUROC', 'Image AUROC', 1.), ('AP', 'Native pixel AP', 1.),
           ('Dice', 'Positive-image Dice', 1.), ('structure_FPR', 'Normal-structure FPR (%)', 100.)]


def main():
    if (BASE / 'final_confirmation_v1/observation_opened.json').exists():
        raise RuntimeError('Development-only readout cannot be revised after confirmation opens')
    efficiency = read_json(BASE / 'inference_efficiency_v1/status.json')['state']
    if not (efficiency.startswith('queued') or efficiency in {'complete', 'failed', 'bounded_stop'}):
        raise RuntimeError('No source plotting during formal inference timings')
    protocol_path = SOURCE / 'protocol.json'; protocol = read_json(protocol_path)
    parent_protocol = ROOT / protocol['upstream_dense_protocol']['path']
    if sha256(parent_protocol) != protocol['upstream_dense_protocol']['sha256']:
        raise RuntimeError('Registered parent protocol changed')
    if sha256(ROOT / 'run_mechanistic_controls.py') != protocol['runner_sha256']:
        raise RuntimeError('Mechanism runner changed')
    for name, expected in protocol['scientific_hashes'].items():
        if sha256(ROOT / 'elresearch' / name) != expected:
            raise RuntimeError('Registered mechanism dependency changed: ' + name)
    dense = read_json(BASE / 'dense_development_complete_cells.json')
    pages = []; pending = []; OUT.mkdir(exist_ok=True)
    for cell in dense['complete_cells']:
        fold, budget = cell['fold'], cell['budget']
        expected = [(family, seed, SOURCE / 'runs' / fold / f'budget_{budget:g}' / f'{family}_seed{seed}')
                    for family in NEW for seed in ([11] if family == 'dinov3_global_normal_distance' and budget == 1 else [11, 22, 33])]
        ready = [(f, s, p) for f, s, p in expected if (p / 'run.json').exists() and
                 (fold == 'internal' or (p / 'target_development.json').exists())]
        if len(ready) != len(expected):
            pending.append({'fold': fold, 'budget': budget, 'complete_new_runs': len(ready), 'required_new_runs': len(expected)})
            continue
        bindings = {}; checks = []; rows = []; support = None; training_ids = {}; post = {}; exports = []

        def bind(path):
            path = Path(path); bindings[str(path.relative_to(ROOT))] = sha256(path)
            return read_json(path) if path.suffix == '.json' else None

        bind(protocol_path); bind(parent_protocol); bind(SOURCE / 'selected_hyperparameters.json')
        original_cell_path = ROOT / cell['cell_path']
        if sha256(original_cell_path) != cell['cell_sha256']: raise RuntimeError('Original complete dense cell changed')
        bind(original_cell_path)
        chosen = []
        for model in cell['models']:
            if sorted(x['seed'] for x in model['fits']) != [11, 22, 33]:
                raise RuntimeError('Original dense controls need all three seeds')
            for fit in model['fits']:
                p = ROOT / fit['run']
                if sha256(p) != fit['run_sha256']: raise RuntimeError('Original dense fit changed')
                chosen.append((model['family'], fit['seed'], p.parent, False))
        chosen += [(f, s, p, True) for f, s, p in ready]
        for family, seed, folder, is_new in chosen:
            run = bind(folder / 'run.json')
            if is_new and run['study_id'] != protocol['study_id']: raise RuntimeError('Wrong mechanism study')
            if (run['seed'], run['training_budget_fraction']) != (seed, budget): raise RuntimeError('Run identity differs')
            if sha256(folder / 'best.pt') != run['checkpoint_sha256']: raise RuntimeError('Checkpoint bytes changed')
            checks.append({'path': str((folder / 'best.pt').relative_to(ROOT)), 'sha256': run['checkpoint_sha256']})
            receipt = run['checkpoint_reload']
            if not receipt['fresh_model'] or max(receipt['logit_max_abs_error'], receipt['sampled_pixel_probability_max_abs_error']) > 2e-5:
                raise RuntimeError('Original fresh-model reload receipt failed')
            if seed in training_ids and set(run['training_ids']) != training_ids[seed]:
                raise RuntimeError('Annotation subsets differ across matched models')
            training_ids[seed] = set(run['training_ids'])
            if family.startswith('dinov3_postlora_'):
                spec = run['model']; parent_path = ROOT / spec['adaptation_parent_run']
                if sha256(parent_path) != spec['adaptation_parent_run_sha256']: raise RuntimeError('Adaptation parent changed')
                parent_run = bind(parent_path)
                if set(parent_run['training_ids']) != set(run['training_ids']) or parent_run['validation_ids'] != run['validation_ids']:
                    raise RuntimeError('Parent/fresh-head annotation identities differ')
                exported = ROOT / spec['adapted_encoder_path']; info = bind(exported / 'encoder_export.json')
                if (sha256(exported / 'model.safetensors') != spec['adapted_weights_sha256'] or
                        sha256(exported / 'config.json') != spec['adapted_config_sha256'] or
                        info['weights_sha256'] != spec['adapted_weights_sha256'] or
                        info['signature']['parent_run_sha256'] != spec['adaptation_parent_run_sha256'] or
                        info['signature']['parent_checkpoint_sha256'] != parent_run['checkpoint_sha256'] or
                        info['merge_max_abs_error'] > 2e-5 or info['fresh_encoder_reload_max_abs_error'] > 2e-5 or
                        info['source_training_probe_id'] not in run['training_ids']):
                    raise RuntimeError('Merged encoder export lineage or original equality proof differs')
                gradient = run['first_gradient_check']
                if gradient['encoder_has_gradient'] or not gradient['heads_have_gradient']:
                    raise RuntimeError('Post-LoRA frozen-encoder/head gradient contract failed')
                signature = {k: spec[k] for k in ['adapted_weights_sha256', 'adaptation_parent_run_sha256', 'head_lr']}
                signature.update(training_ids=sorted(run['training_ids']), validation_ids=run['validation_ids'],
                                 trainable_parameters=run['trainable_parameters'])
                if seed in post and post[seed] != signature: raise RuntimeError('Post-LoRA reference pair is not matched')
                post[seed] = signature
                exports.append({'seed': seed, 'reference': spec['reference'], 'encoder_export': str((exported / 'encoder_export.json').relative_to(ROOT)),
                                'weights_sha256': spec['adapted_weights_sha256'], 'original_merge_error': info['merge_max_abs_error']})
            if is_new and run['model'].get('reference'):
                recipe = bind(folder / 'normal_bank_recipe.json')
                if recipe['encoder']['weights_sha256'] != run['encoder']['weights_sha256']:
                    raise RuntimeError('Reference memory uses a different encoder')
                if any(row['id'] not in run['training_ids'] for row in recipe['rows']):
                    raise RuntimeError('Memory includes a nontraining image')
                if 'selected_training_ids' in recipe and set(recipe['selected_training_ids']) != set(run['training_ids']):
                    raise RuntimeError('Canonical memory recipe training pool differs')
                if 'payload' in recipe:
                    payload = recipe['payload']
                    if sha256(ROOT / payload['path']) != payload['sha256']: raise RuntimeError('Canonical normal memory changed')
                    checks.append(payload)
            evaluation = run if fold == 'internal' else bind(folder / 'target_development.json')
            metrics = run['validation_metrics'] if fold == 'internal' else evaluation['metrics']
            local = run['validation_localization'] if fold == 'internal' else evaluation['localization']
            pixels = local['per_image']; ids = [p['id'] for p in pixels]
            if fold == 'internal' and ids != run['validation_ids']: raise RuntimeError('Native/source row order differs')
            if set(ids) & set(run['training_ids']): raise RuntimeError('Evaluation is in the training subset')
            if fold != 'internal' and set(ids) & set(run['validation_ids']): raise RuntimeError('Held evaluation is in validation')
            annotation = [(p['id'], p['known_pixels'], p['crack_pixels'], p['negative_image_normal_structure_pixels']) for p in pixels]
            if support is None: support = annotation
            elif annotation != support: raise RuntimeError('Native annotation support differs across models')
            predictions = folder / ('validation_predictions.npz' if fold == 'internal' else 'target_development_predictions.npz')
            bind(predictions)
            with np.load(predictions, allow_pickle=False) as z:
                if z['ids'].tolist() != ids: raise RuntimeError('Original image predictions order differs')
                np.testing.assert_array_equal(z['y'], [int(p['crack_positive']) for p in pixels])
                actual = classification(z['y'], z['logits'], run['source_logit_threshold'])
                for key, value in metrics.items():
                    if value is None:
                        if actual[key] is not None: raise RuntimeError('Null image metric differs')
                    elif not np.isclose(actual[key], value, atol=2e-12, rtol=0): raise RuntimeError('Saved image metric differs: ' + key)
            positive_dice = []; structure_pixels = 0; structure_false = 0
            for p in pixels:
                tp = p['predicted_crack_pixels'] - p['false_positive_pixels']; denom = p['predicted_crack_pixels'] + p['crack_pixels']
                dice = 2 * tp / denom if denom else 1.
                if not np.isclose(dice, p['dice'], atol=2e-12, rtol=0): raise RuntimeError('Native integer counts disagree with Dice')
                if p['crack_positive']: positive_dice.append(dice)
                if p['negative_image_normal_structure_pixels'] is not None:
                    structure_pixels += p['negative_image_normal_structure_pixels']
                    structure_false += p['negative_image_normal_structure_false_positive_pixels']
            structure_fpr = structure_false / structure_pixels if structure_pixels else None
            if not np.isclose(statistics.mean(positive_dice), local['mean_positive_image_dice'], atol=2e-12, rtol=0):
                raise RuntimeError('Native positive-image Dice mean differs')
            if structure_fpr is not None and not np.isclose(structure_fpr, local['pooled_negative_structure_false_positive_fraction'], atol=2e-12, rtol=0):
                raise RuntimeError('Native normal-structure FPR differs')
            rows.append({'family': family, 'seed': seed, 'training_masks': run['training_images'],
                         'validation_masks': run['validation_images'], 'evaluation_images': len(pixels),
                         'AUROC': metrics['auc'], 'AP': local['pooled_pixel_average_precision'],
                         'Dice': local['mean_positive_image_dice'], 'structure_FPR': structure_fpr,
                         'image_recall': metrics['recall'], 'image_review_fraction': metrics['review_rate'],
                         'predicted_crack_pixels': sum(p['predicted_crack_pixels'] for p in pixels),
                         'positive_images_with_empty_prediction': sum(p['crack_positive'] and p['predicted_crack_pixels'] == 0 for p in pixels),
                         'positive_images': local['positive_images'], 'source_run': str((folder / 'run.json').relative_to(ROOT)),
                         'source_threshold': run['source_logit_threshold'],
                         'pixel_cutoff_semantics': 'source99th-percentile noncrack response, not crack posterior' if family == 'dinov3_global_normal_distance' else 'learned crack softmax >0.5'})
        families = [m['family'] for m in cell['models']] + NEW
        summaries = []
        for family in families:
            rr = sorted((r for r in rows if r['family'] == family), key=lambda r: r['seed'])
            expected_seeds = [11] if family == 'dinov3_global_normal_distance' and budget == 1 else [11, 22, 33]
            if [r['seed'] for r in rr] != expected_seeds: raise RuntimeError('Incomplete declared seed set')
            summaries.append({'family': family, 'fits': len(rr), 'mean': {k: statistics.mean(r[k] for r in rr) for k, _, _ in METRICS},
                              'sample_SD': {k: statistics.stdev(r[k] for r in rr) if len(rr) > 1 else None for k, _, _ in METRICS}})
        pairs = []
        for seed in [11, 22, 33]:
            first = next(r for r in rows if r['family'] == NEW[2] and r['seed'] == seed)
            second = next(r for r in rows if r['family'] == NEW[3] and r['seed'] == seed)
            pairs.append({'seed': seed, 'difference_reference_minus_no_reference': {k: second[k] - first[k] for k, _, _ in METRICS},
                          'no_reference_predicted_pixels': first['predicted_crack_pixels'],
                          'reference_predicted_pixels': second['predicted_crack_pixels']})
        identity = {'fold': fold, 'budget': budget, 'builder_sha256': sha256(__file__),
                    'input_bindings': [{'path': p, 'sha256': h} for p, h in sorted(bindings.items())],
                    'complete_family_order': families, 'rule': 'All original dense systems and all four registered mechanism arms with every required seed; descriptive source-only outcomes.'}
        directory = OUT / digest(identity)[:20]; result_path = directory / 'results.json'
        if result_path.exists():
            result = read_json(result_path)
            if result['identity'] != identity: raise RuntimeError('Prior complete-cell snapshot differs')
            for asset in result['figures']:
                if sha256(ROOT / asset['path']) != asset['sha256']: raise RuntimeError('Prior figure changed')
        else:
            directory.mkdir(exist_ok=False); assets = []
            style = {'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
            with plt.rc_context(style):
                fig, axes = plt.subplots(1, 4, figsize=(9.2, 6.2), sharey=True)
                fig.subplots_adjust(left=.30, right=.98, top=.77, bottom=.31, wspace=.4)
                for ax, (key, title, scale) in zip(axes, METRICS):
                    for y, summary in enumerate(summaries):
                        color = '#7955a8' if summary['family'].startswith('dinov3_postlora') else ('#b97f31' if summary['family'] == NEW[1] else '#3c7385')
                        value = summary['mean'][key] * scale; sd = summary['sample_SD'][key]
                        ax.errorbar(value, y, xerr=sd * scale if sd is not None else None, fmt='D', ms=3.5, color=color, capsize=2.4, lw=1)
                        for off, marker, r in zip([-.09, 0., .09], ['o', '^', 's'], sorted((r for r in rows if r['family'] == summary['family']), key=lambda r: r['seed'])):
                            ax.scatter(r[key] * scale, y+off, marker=marker, s=13, color=color, alpha=.4)
                    if key == 'AUROC': ax.set_xlim(0, 1.02); ax.axvline(.5, ls=':', color='#a4b1b7', lw=.8)
                    else: ax.set_xlim(left=0)
                    ax.set_title('Normal-structure\nFPR (%)' if key == 'structure_FPR' else title,
                                 fontsize=8, loc='left', pad=12); ax.grid(axis='x', color='#e5ecef', lw=.6)
                    ax.set_yticks(range(len(families)), [NAMES[f] for f in families]); ax.set_ylim(len(families)-.45, -.55)
                    ax.tick_params(axis='y', length=0, pad=9)
                    for side in ['top', 'right', 'left']: ax.spines[side].set_visible(False)
                fig.text(.06, .94, f"Complete mechanism controls · {fold} · {100*budget:g}% source training masks", fontsize=11, weight='bold', color='#193f4d')
                fig.text(.06, .87, f"{rows[0]['training_masks']} training masks + {rows[0]['validation_masks']} validation masks; upstream UCF574 exposure is additional", fontsize=8.2)
                notes = ['Diamonds/bars: mean ± sample SD. Faint points retain every declared fitted seed/subset; no confidence interval.',
                         '* Normal distance uses a source noncrack-FPR cutoff, not learned crack posterior0.5; Dice conventions differ.',
                         'Native pooled pixel AP is a saved cohort-level point; pixels are not independent replicates.',
                         'Post-LoRA arms share the same merged encoder, annotation identities and fresh-head architecture/initialization.',
                         'Parent LoRA adaptation plus a fresh decoder adds a second training stage; equal labels do not mean equal compute.',
                         'All rows are source/development observations. No target fitting, new inference or independent confirmation.']
                for i, note in enumerate(notes): fig.text(.06, .23-i*.032, note, fontsize=7, color='#526e78')
                for ext in ['png', 'pdf', 'svg']:
                    path = directory / ('all_completed_arms.'+ext); fig.savefig(path, dpi=250, facecolor='white'); assets.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
                plt.close(fig)
                fig, axes = plt.subplots(1, 4, figsize=(8.5, 3.6)); fig.subplots_adjust(left=.08, right=.98, top=.68, bottom=.39, wspace=.55)
                for ax, (key, title, scale) in zip(axes, METRICS):
                    values = [p['difference_reference_minus_no_reference'][key] * scale for p in pairs]
                    ax.axhline(0, color='#a5b3bb', ls='--', lw=.8)
                    ax.scatter([0, 1, 2], values, c='#7955a8', s=24); ax.scatter([3.2], [statistics.mean(values)], marker='D', c='#253c4a', s=25)
                    ax.set_xticks([0, 1, 2, 3.2], ['11', '22', '33', 'Mean']); ax.set_title('Δ '+('Structure FPR (pp)' if key == 'structure_FPR' else title), fontsize=8)
                    ax.set_xlim(-.5, 3.7); ax.grid(axis='y', color='#e7edef', lw=.6)
                    for side in ['top', 'right']: ax.spines[side].set_visible(False)
                fig.text(.08, .92, 'Post-LoRA: reference minus matched fresh decoder', fontsize=11, weight='bold', color='#193f4d')
                fig.text(.08, .80, f"{fold}, {100*budget:g}% masks · every paired seed; descriptive differences, no CI or p-value", fontsize=8)
                for i, note in enumerate(['Positive AUROC/AP/Dice differences favor reference; negative structure-FPR differences favor reference.',
                                           'FPR differences displayed in percentage points; JSON/CSV retain raw probability-ratio differences.',
                                           'All fitted pairs retained, including empty predicted masks. Reused development is not independent confirmation.']):
                    fig.text(.06, .24-i*.065, note, fontsize=7, color='#526e78')
                for ext in ['png', 'pdf', 'svg']:
                    path = directory / ('all_postlora_paired_seeds.'+ext); fig.savefig(path, dpi=250, facecolor='white'); assets.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
                plt.close(fig)
            csv_path = directory / 'every_original_fit.csv'
            with csv_path.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            result = {'at': now(), 'identity': identity, 'fold': fold, 'budget': budget,
                      'all_original_fit_records': rows, 'all_model_summaries': summaries,
                      'all_three_postlora_paired_seeds': pairs, 'adapted_encoder_lineage_checks': exports,
                      'existing_checkpoint_and_payload_hash_checks': checks, 'figures': assets,
                      'csv': {'path': str(csv_path.relative_to(ROOT)), 'sha256': sha256(csv_path)},
                      'native_AP_recomputed': False,
                      'integrity_scope': 'Saved image-score metrics and integer native Dice/structure-FPR rechecked. Pooled native AP is preserved from original whole-native evaluation, not reconstructed from per-image AP.',
                      'new_model_fitting_or_inference': False, 'new_statistical_test': False, 'final_confirmation_access': False}
            save_json(result_path, result)
        pages.append({'fold': fold, 'budget': budget, 'path': str(result_path.relative_to(ROOT)), 'sha256': sha256(result_path)})
    index = {'at': now(), 'builder_sha256': sha256(__file__), 'complete_cells': pages, 'pending_cells': pending,
             'planned_cells': 10, 'rule': 'No cell comparison until all original controls and all four mechanism arms have their declared complete seed sets.'}
    save_json(OUT / 'index.json', index)
    cards = []
    for page in pages:
        result = read_json(ROOT / page['path']); folder = (ROOT / page['path']).parent
        image = next(a for a in result['figures'] if a['path'].endswith('all_completed_arms.png'))
        paired = next(a for a in result['figures'] if a['path'].endswith('all_postlora_paired_seeds.png'))
        cards.append(f'<section><h2>{html.escape(page["fold"])} · {100*page["budget"]:g}% training masks</h2><img src="{Path(image["path"]).parent.name}/{Path(image["path"]).name}"><img src="{Path(paired["path"]).parent.name}/{Path(paired["path"]).name}"><p><a href="{folder.name}/results.json">Every saved endpoint, lineage check and paired seed</a> · <a href="{folder.name}/every_original_fit.csv">Every original fit</a></p></section>')
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Completed mechanism controls</title><style>body{max-width:1150px;margin:48px auto;font:17px/1.6 system-ui;color:#1d414e}img{width:100%}section{margin:42px 0}a{color:#287186}</style><h1>Complete source mechanism controls</h1><p>All original dense models plus all four mechanism arms appear only after each declared seed set is complete. Post-LoRA means a frozen merged parent encoder plus a new decoder; parent adaptation adds training work. Distance and learned-softmax cutoffs differ. All data here are observed source/development, with no new model inference or independent confirmation.</p>' + ''.join(cards) + '<h2>Pending complete cells</h2><pre>' + html.escape(str(pending)) + '</pre></html>'
    (OUT / 'report.html').write_text(document)
    print('Complete mechanism cells', len(pages), '/10; pending', len(pending), '; no new model inference')


if __name__ == '__main__':
    main()
