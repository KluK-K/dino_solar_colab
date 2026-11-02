"""Complete retrospective classical controls: reload, all cells and paired limits.

This analysis never fits a classifier or reads a confirmation model output.
The classical runner and scientific closure remain unchanged.
"""
import csv
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from elresearch.common import ROOT, read_json, save_json, sha256, now
from elresearch.metrics import classification
from elresearch.publication_splits import fold_records
from elresearch.publication_statistics import paired_seed_group_bootstrap
from run_classical_controls import load_banks

BASE = ROOT/'artifacts/publication_20261017'
SOURCE = BASE/'classical_controls_v1'
OUTPUT = BASE/'classical_development_analysis_v1'
FAMILIES = [r+'_'+c for r in ['hog_stats', 'dinov3_cls']
            for c in ['logistic', 'linear_svm', 'rbf_svm']]


def main():
    if (BASE/'final_confirmation_v1/observation_opened.json').exists():
        raise RuntimeError('This synthesis precedes independent confirmation')
    settings = read_json(SOURCE/'protocol.json')
    if read_json(SOURCE/'status.json')['state'] != 'complete':
        raise RuntimeError('All registered classical fits must finish')
    stats = read_json(BASE/'statistics_plan_v2.json')
    plan = {'version': 1, 'kind': 'Retrospective complete classical-development synthesis',
        'source_protocol_sha256': sha256(SOURCE/'protocol.json'),
        'image_protocol_sha256': sha256(BASE/'scdd_development_v1/protocol.json'),
        'analysis_code_sha256': sha256(__file__),
        'statistics_code_sha256': sha256(ROOT/'elresearch/publication_statistics.py'),
        'all_classical_families': FAMILIES, 'comparison': 'LoRA minus every classical family on every complete held-source cell',
        'repetitions': stats['repetitions'], 'seed': stats['seed'],
        'development_results_previously_observed': True,
        'scope': 'Reload and verify every186 selected classical pipeline; no refit, encoder forward, target calibration or confirmation score.100%-budget deterministic seed records are one optimizer outcome, not3 repetitions.'}
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT/'plan.json').exists() and read_json(OUTPUT/'plan.json') != plan:
        raise RuntimeError('Preserve previous analysis plan/code before an amendment')
    if not (OUTPUT/'plan.json').exists(): save_json(OUTPUT/'plan.json', plan)
    expected = 0; cells = []; comparisons = []; verifications = []
    training, validation, _ = fold_records(settings['split_manifest'], 'internal')
    banks = load_banks(settings, 'scdd', training+validation)

    def verify(folder, bank, validation, target=None):
        run = read_json(folder/'run.json')
        if sha256(folder/'model.joblib') != run['checkpoint_sha256']:
            raise RuntimeError('Saved model hash mismatch')
        if not run['source_reload_verified'] or run['solver_attempts'][-1]['convergence_warnings']:
            raise RuntimeError('Unverified or unconverged final classifier')
        if bank.feature_sha256 != run['feature_bank_sha256']:
            raise RuntimeError('Feature source identity changed')
        model = joblib.load(folder/'model.joblib')
        outputs = {}
        for name, rows in [('validation', validation), ('target_development', target)]:
            if rows is None: continue
            with np.load(folder/f'{name}_predictions.npz', allow_pickle=False) as saved:
                values = {k:saved[k].copy() for k in saved.files}
            np.testing.assert_array_equal(values['ids'], [r['id'] for r in rows])
            np.testing.assert_array_equal(values['y'], [r['target'] for r in rows])
            with threadpool_limits(limits=2):
                scores = np.asarray(model.decision_function(bank.get(rows)), dtype=np.float64)
            np.testing.assert_array_equal(scores, values['logits'])
            metrics = classification(values['y'], scores, run['source_calibrated_margin_threshold'])
            stored = run['validation_metrics'] if name=='validation' else read_json(folder/'target_development.json')['metrics']
            if metrics != stored: raise RuntimeError('Recomputed metrics differ')
            outputs[name] = {**values, 'seed': run['seed'], 'threshold': run['source_calibrated_margin_threshold']}
        verifications.append({'run':str((folder/'run.json').relative_to(ROOT)),
            'run_sha256':sha256(folder/'run.json'), 'checkpoint_sha256':run['checkpoint_sha256'],
            'prediction_sha256':{name:sha256(folder/f'{name}_predictions.npz') for name in outputs},
            'exact_reload': True, 'final_solver_convergence_warning': False})
        return run, outputs

    for fold in settings['split_manifest']['folds']:
        training, validation, target = fold_records(settings['split_manifest'], fold['name'])
        for budget in fold['budgets']:
            cell={'fold':fold['name'], 'training_fraction':budget, 'models':[]}
            chosen = {}
            for family in FAMILIES:
                values=[]; prediction_rows=[]
                for seed in [11,22,33]:
                    folder=SOURCE/'runs'/fold['name']/f'budget_{budget:g}'/f'{family}_seed{seed}'
                    representation='hog_stats' if family.startswith('hog_stats_') else 'dinov3_cls'
                    run, outputs=verify(folder,banks[representation],validation,target or None)
                    expected+=1
                    image_folder=BASE/'scdd_development_v1/runs'/fold['name']/f'budget_{budget:g}'/f'dinov3_lora_seed{seed}'
                    image=read_json(image_folder/'run.json')
                    if set(run['training_ids'])!=set(image['training_ids']) or run['validation_ids']!=image['validation_ids']:
                        raise RuntimeError('Classical/neural supervision identities differ')
                    name='target_development' if target else 'validation'
                    stored=read_json(folder/'target_development.json')['metrics'] if target else run['validation_metrics']
                    values.append({'seed':seed,'metrics':stored,'training_images':run['training_images'],
                        'validation_images':run['validation_images'],'run_sha256':sha256(folder/'run.json')})
                    prediction_rows.append(outputs[name])
                if budget==1.:
                    for row in prediction_rows[1:]:
                        np.testing.assert_array_equal(row['logits'],prediction_rows[0]['logits'])
                cell['models'].append({'family':family,'seeds':values,
                    'mean':{key:float(np.mean([r['metrics'][key] for r in values])) for key in ['auc','average_precision','recall','review_rate']},
                    'sample_sd':{key:float(np.std([r['metrics'][key] for r in values],ddof=1)) for key in ['auc','average_precision','recall','review_rate']},
                    'optimizer_repetitions':1 if budget==1. else None,
                    'variability':'At partial budgets, variation reflects selected training subsets; at100% deterministic predictions are identical.'})
                chosen[family]=prediction_rows
            cells.append(cell)
            if target:
                lora=[]
                for seed in [11,22,33]:
                    folder=BASE/'scdd_development_v1/runs'/fold['name']/f'budget_{budget:g}'/f'dinov3_lora_seed{seed}'
                    run=read_json(folder/'run.json');target_meta=read_json(folder/'target_development.json')
                    if sha256(folder/'target_development_predictions.npz')!=target_meta['prediction_sha256']:
                        raise RuntimeError('LoRA saved predictions changed')
                    with np.load(folder/'target_development_predictions.npz',allow_pickle=False) as saved:
                        lora.append({**{k:saved[k].copy() for k in ['ids','y','groups','logits']},
                                     'seed':seed,'threshold':run['source_calibrated_logit_threshold']})
                for family in FAMILIES:
                    comparisons.append({'fold':fold['name'],'training_fraction':budget,
                        'first':family,'second':'dinov3_lora',
                        'bootstrap':paired_seed_group_bootstrap(chosen[family],lora,plan['repetitions'],plan['seed'])})
                print('Verified source cell',fold['name'],budget,flush=True)
    if expected!=180 or len(cells)!=10 or len(comparisons)!=36:
        raise RuntimeError('Do not publish a partial classical comparison')
    # Reload the six source-selected UCF pipelines as well; cached encoders only.
    from elresearch.data import records
    training,validation=records('ucf','train'),records('ucf','validation')
    source_banks=load_banks(settings,'ucf',training+validation)
    selection=read_json(SOURCE/'selected_hyperparameters.json')
    for row in selection['selected']:
        verify(ROOT/row['source_model_path'],source_banks[row['specification']['representation']],validation)
    trials=list((SOURCE/'calibration_trials').glob('*/run.json'))
    if len(trials)!=70 or any(read_json(p)['solver_attempts'][-1]['convergence_warnings'] for p in trials):
        raise RuntimeError('Require all70 converged source-grid results')
    result={'at':now(),'status':'complete','plan_sha256':sha256(OUTPUT/'plan.json'),
        'verified_selected_model_count':len(verifications),'verified_source_calibration_trials':70,
        'cells':cells,'comparisons':comparisons,'verifications':verifications,
        'source_selection':selection,'confirmation_scores_opened':False,
        'limits':['Retrospective exploratory development comparisons; no multiplicity-adjusted superiority claim.',
                  'Intervals condition on fitted models and provided capture families, not verified physical modules.',
                  'Classical full-budget deterministic records are not independent optimizer replications.',
                  'Seven-C and three-RBF-width grids differ from two-rate neural calibration; equal tuning costs not claimed.',
                  'Same training identities do not imply identical losses or optimization procedures.']}
    save_json(OUTPUT/'results.json',result)
    rows=[]
    for cell in cells:
        for model in cell['models']:
            rows.append({'fold':cell['fold'],'training_fraction':cell['training_fraction'],
                'model':model['family'],**model['mean'],
                'auc_sample_sd':model['sample_sd']['auc'],'optimizer_repetitions':model['optimizer_repetitions']})
    with (OUTPUT/'all_classical_cells.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('COMPLETE',len(verifications),'exact classical reloads;',len(comparisons),'exploratory paired contrasts',flush=True)


if __name__=='__main__':main()
