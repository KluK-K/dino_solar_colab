"""Queue a separate original-plan ELPV supplement after the source cost benchmark."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time
import traceback
from importlib.metadata import version

from elresearch.common import ROOT, read_json, save_json, sha256, digest, now
from elresearch import elpv_supplement as study

BASE = ROOT/'artifacts/publication_20261017'
FOLDER = BASE/'elpv_review_supplement_v1'


def protocol():
    rows, folds = study.checked_development()
    files = [ROOT/'elresearch'/name for name in ['elpv_supplement.py', 'encoders.py', 'metrics.py', 'common.py', 'data.py']]
    files += list(study.inputs().values()) + [ROOT/'config.json', Path(__file__)]
    value = {'version': 1, 'task': 'ELPV expert_label>0 requires review; supplementary development-only nested CV',
        'development_images': 1968, 'course_test_sealed': 656, 'planned_runs': 24, 'outer_folds': 4, 'inner_folds': 3,
        'inner_grid_fits': 480, 'outer_refits': 24,
        'runtime_versions': {name: version(name) for name in ['numpy', 'scikit-learn', 'torch', 'transformers', 'Pillow']},
        'classifiers': ['logistic', 'linear_svm', 'rbf_svm'], 'features': ['original course1806 HOG/statistics', 'official frozen DINOv3-S CLS384 at336'],
        'source_bindings': [{'path': str(p), 'sha256': sha256(p)} for p in files],
        'sample_ids_sha256': digest([r['id'] for r in rows]),
        'stage': 'After complete inference benchmark; no competition with own timed jobs. No final crack-confirmation scores used.',
        'limits': ['No DINOv2; no ELPV656 test reads.', 'Original course fold identities reused; hyperparameters/scaler newly nested inside training folds.',
            'Four CV folds are not four independent physical populations. No claim of outperforming Deitsch paper scores.']}
    value['study_id'] = digest(value)[:20]
    return value


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--wait-for-efficiency', action='store_true'); parser.add_argument('--register-only', action='store_true'); args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    with (FOLDER/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proposed = protocol(); path = FOLDER/'protocol.json'
        if path.exists() and proposed != read_json(path): raise RuntimeError('Registered supplementary study changed')
        if not path.exists():
            save_json(path, proposed); snapshots = FOLDER/'source_snapshot'; snapshots.mkdir()
            for number, value in enumerate(proposed['source_bindings']):
                p = Path(value['path'])
                if p.suffix == '.py': shutil.copy2(p, snapshots/f'{number}_{p.name}')
        if args.register_only: print('Registered ELPV development-only supplement', proposed['study_id']); return
        if (FOLDER/'status.json').exists() and read_json(FOLDER/'status.json')['state'] == 'complete': return
        while True:
            state = read_json(BASE/'inference_efficiency_v1/status.json')
            if state['state'] == 'complete': break
            if not args.wait_for_efficiency: raise RuntimeError('Finish the complete source efficiency benchmark first')
            save_json(FOLDER/'status.json', {'at': now(), 'pid': os.getpid(), 'state': 'queued_after_efficiency', 'completed_runs': 0, 'planned_runs': 24})
            time.sleep(30)
        if shutil.disk_usage(ROOT).free < 2*1024**3: raise RuntimeError('Preserve existing artifacts; insufficient space for supplementary features')
        if proposed != protocol(): raise RuntimeError('Supplementary source bindings changed while queued')
        rows, folds = study.checked_development()
        with (BASE/'gpu_work_queue.lock').open('a') as gpu:
            fcntl.flock(gpu, fcntl.LOCK_EX)
            X = study.extract_cls(rows, FOLDER)
        study.fit_all(rows, folds, X, FOLDER)
        save_json(FOLDER/'status.json', {'at': now(), 'state': 'complete', 'completed_runs': 24, 'planned_runs': 24,
            'course_test_opened': False, 'scope': 'Supplementary nested development CV, not crack confirmation'})
        save_json(FOLDER/'verification.json', study.verify_complete(FOLDER))


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        FOLDER.mkdir(parents=True, exist_ok=True)
        save_json(FOLDER/'failure.json', {'at': now(), 'error': str(error), 'traceback': traceback.format_exc()})
        raise
