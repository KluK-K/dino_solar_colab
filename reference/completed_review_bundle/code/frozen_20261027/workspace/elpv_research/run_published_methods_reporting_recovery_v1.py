"""Registered recovery for one missing dense-result reporting metadata field.

The original runner/protocol, scientific implementations, fits and selection
rules stay unchanged. This entrypoint completes only the child-settings schema
passed to the original dense trainer, which owns optimizer/RNG restoration.
"""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).absolute().parent
RECOVERY = ROOT / 'artifacts/publication_20261017/operational_recovery_v1/selection_metadata_2026-10-27T00-31-35.243048+00-00'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def reporting_settings(settings, selection):
    """Return unchanged computational settings plus the original description."""
    if 'selection' in settings and settings['selection'] != selection:
        raise RuntimeError('Do not replace an existing selection rule')
    result = {**settings, 'selection': selection}
    if {key: value for key, value in result.items() if key != 'selection'} != {
            key: value for key, value in settings.items() if key != 'selection'}:
        raise RuntimeError('Recovery changed a computational setting')
    return result


def main():
    amendment_path = RECOVERY / 'amendment.json'
    amendment = read(amendment_path)
    for binding in amendment['immutable_bindings']:
        if sha(ROOT / binding['path']) != binding['sha256']:
            raise RuntimeError('Registered reporting recovery binding changed: ' + binding['path'])
    if sha(Path(__file__).absolute()) != amendment['recovery_entrypoint']['sha256']:
        raise RuntimeError('Registered recovery entrypoint changed')
    import run_published_methods as original
    original_train = original.train_dense
    published = read(ROOT / 'artifacts/publication_20261017/published_methods_v1/protocol.json')
    selection = read(ROOT / 'artifacts/publication_20261017/dense_development_v1/protocol.json')['selection']
    if selection != amendment['reporting_selection_value']:
        raise RuntimeError('Original reporting description changed')
    names = {'dinov3_frozen_ucf20', 'dinov3_lora_ucf20', 'dinov3_reference_ucf20'}
    first = amendment['failed_fit_resume_contract']

    def completed_reporting_train(settings, spec, source_train, source_validation, seed, budget, folder):
        folder = Path(folder)
        if (settings['study_id'] != published['study_id'] or spec['name'] not in names or
                seed not in [11, 22, 33] or budget != .2):
            raise RuntimeError('Recovery is restricted to the original nine matched controls')
        child = reporting_settings(settings, selection)
        if set(child) - set(settings) != ({'selection'} if 'selection' not in settings else set()):
            raise RuntimeError('Recovery added more than the one reporting field')
        is_failed_fit = str(folder.relative_to(ROOT)) == first['folder']
        if is_failed_fit:
            for name in ['started.json', 'history.json', 'best.pt']:
                if sha(folder / name) != first['original_files'][name]['sha256']:
                    raise RuntimeError('Failed fit identity/history/best checkpoint changed')
        event = RECOVERY / 'adapter_events' / (now().replace(':', '-') + '__' + folder.name)
        save_new(event / 'before.json', {
            'at': now(), 'amendment_sha256': sha(amendment_path),
            'run_directory': str(folder.relative_to(ROOT)), 'model': spec['name'], 'seed': seed,
            'only_added_setting': 'selection', 'reporting_selection_value': selection,
            'original_settings_unchanged': settings == {key: child[key] for key in settings},
            'optimizer_RNG_restore_is_original_trainer': True,
            'failed_fit_resume_contract_applies': is_failed_fit})
        result = original_train(child, spec, source_train, source_validation, seed, budget, folder)
        if is_failed_fit:
            if (result['completed_rounds'] != first['saved_round'] or
                    result['optimizer_updates'] != first['optimizer_updates']):
                raise RuntimeError('Unexpected optimizer work after already-triggered original early stop')
            for name in ['started.json', 'history.json', 'best.pt']:
                if sha(folder / name) != first['original_files'][name]['sha256']:
                    raise RuntimeError('Original failed-fit history/best state was not preserved')
        save_new(event / 'after.json', {
            'at': now(), 'run': {'path': str((folder / 'run.json').relative_to(ROOT)),
                                'sha256': sha(folder / 'run.json')},
            'completed_rounds': result['completed_rounds'], 'optimizer_updates': result['optimizer_updates'],
            'selection_metadata_matches_upstream': result['selection'] == selection,
            'original_trainer_returned_normally': True,
            'failed_fit_history_and_best_exactly_preserved': is_failed_fit,
            'additional_optimizer_updates_for_failed_fit': 0 if is_failed_fit else None,
            'new_independent_fit_or_confirmation': False})
        return result

    original.train_dense = completed_reporting_train
    print('REGISTERED REPORTING-ONLY RECOVERY', sha(amendment_path), flush=True)
    original.main()


if __name__ == '__main__':
    main()
