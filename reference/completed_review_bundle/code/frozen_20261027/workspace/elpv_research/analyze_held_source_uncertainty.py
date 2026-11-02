"""Apply the existing registered cluster-bootstrap plan to development predictions."""
import os
import json
import hashlib
import shutil
import numpy as np

from elresearch.common import ROOT,read_json,save_json,sha256,digest,now
from elresearch.publication_statistics import paired_seed_group_bootstrap


BASE=ROOT/'artifacts/publication_20261017'


def main():
    folder=BASE/'held_source_uncertainty_v1';folder.mkdir(exist_ok=True)
    plan_path=BASE/'statistics_plan_v2.json';plan=read_json(plan_path)
    if sha256(ROOT/'elresearch/publication_statistics.py')!=plan['code_sha256']:
        raise RuntimeError('Registered bootstrap implementation changed')
    source=BASE/'held_source_development_complete_cells.json';raw=source.read_bytes()
    source_sha=hashlib.sha256(raw).hexdigest();snapshot=folder/'input_snapshots'/(source_sha+'.json')
    snapshot.parent.mkdir(exist_ok=True)
    if snapshot.exists() and sha256(snapshot)!=source_sha:raise RuntimeError('Archived development snapshot changed')
    if not snapshot.exists():snapshot.write_bytes(raw)
    cells=json.loads(raw)['complete_cells'];analysis_sha=sha256(__file__)
    code_archive=folder/'analysis_snapshots';code_archive.mkdir(exist_ok=True)
    if not (code_archive/(analysis_sha+'.py')).exists():shutil.copy2(__file__,code_archive/(analysis_sha+'.py'))
    split=read_json(BASE/'scdd_development_v1/protocol.json')['split_manifest']
    folds={f['name']:f for f in split['folds']}
    baselines=['dinov3_frozen','resnet18_full','resnet50_full','efficientnet_b0_full']
    records=[];planned=len(cells)*len(baselines)
    for cell in cells:
        support={x['family']:x for x in cell['models']};predictions={}
        for family in baselines+['dinov3_lora']:
            per_seed=[]
            for seedrow in support[family]['seeds']:
                run_path=ROOT/seedrow['run'];run=read_json(run_path)
                path=run_path.parent/'target_development_predictions.npz'
                if sha256(path)!=seedrow['prediction_sha256'] or sha256(run_path)!=seedrow['run_sha256']:
                    raise RuntimeError('Development predictions/source model changed')
                with np.load(path,allow_pickle=False) as f:
                    value={key:f[key].copy() for key in ['ids','y','groups','logits']}
                np.testing.assert_array_equal(value['ids'],folds[cell['fold']]['target_development_ids'])
                value.update(seed=seedrow['seed'],threshold=run['source_calibrated_logit_threshold'])
                per_seed.append(value)
            predictions[family]=per_seed
        for baseline in baselines:
            key=digest({'cell':cell,'first':baseline,'second':'dinov3_lora','plan_sha256':sha256(plan_path),
                'analysis_code_sha256':analysis_sha})
            destination=folder/(key+'.json')
            save_json(folder/'status.json',{'at':now(),'pid':os.getpid(),'state':'running','completed_comparisons':len(records),
                'planned_comparisons':planned,'current':{'fold':cell['fold'],'budget':cell['training_fraction'],'baseline':baseline}})
            if destination.exists():result=read_json(destination)
            else:
                contrast=paired_seed_group_bootstrap(predictions[baseline],predictions['dinov3_lora'],
                    repetitions=plan['repetitions'],seed=plan['seed'])
                result={'at':now(),'fold':cell['fold'],'training_fraction':cell['training_fraction'],'first':baseline,
                    'second':'dinov3_lora','source_cell':cell,'bootstrap':contrast,'input_key':key,
                    'statistics_plan_sha256':sha256(plan_path),'analysis_code_sha256':analysis_sha,
                    'archived_source_snapshot':{'path':str(snapshot.relative_to(ROOT)),'sha256':source_sha},
                    'scope':'Exploratory development; conditional acquisition-family interval, not adjusted significance or a physical-module CI'}
                save_json(destination,result)
            records.append(result)
            ci=result['bootstrap']['metrics']['auroc']
            print('DEVELOPMENT CONTRAST',cell['fold'],cell['training_fraction'],'LoRA minus',baseline,
                'AUROC difference',round(ci['mean_seed_difference'],4),'cluster CI',ci['acquisition_family_percentile_ci95'],flush=True)
    save_json(BASE/'held_source_paired_uncertainty.json',{'at':now(),'source_snapshot_sha256':source_sha,
        'archived_source_snapshot':{'path':str(snapshot.relative_to(ROOT)),'sha256':source_sha},
        'statistics_plan_sha256':sha256(plan_path),'comparisons':records,
        'scope':'All complete development cells and all four contrasts in the prior plan; final confirmation remains sealed'})
    save_json(folder/'status.json',{'at':now(),'pid':os.getpid(),'state':'complete','completed_comparisons':len(records),'planned_comparisons':planned})


if __name__=='__main__':main()
