"""Runtime inventory and safe access check; never fetch model weights."""
import html
import platform
import importlib.metadata
from pathlib import Path

from .common import ROOT, config, save_json, read_json, choose_device, now

def preflight(check_access=False):
    import torch
    from huggingface_hub import get_token, hf_hub_url
    import requests
    spec = config()['models']['dinov3_small']
    token = get_token()
    result = {'created_at': now(), 'platform': platform.platform(), 'machine': platform.machine(),
              'python': platform.python_version(), 'device': str(choose_device()),
              'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available(),
              'versions': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'transformers', 'huggingface-hub', 'numpy', 'scikit-learn']},
              'huggingface_token_present': bool(token), 'model': spec['id'],
              'access': 'not_checked', 'weights_downloaded_by_preflight': False,
              'disabled_encoders': config().get('disabled_encoders', [])}
    if check_access:
        # Config is tiny; only HEAD is needed. Credentials never appear in output.
        url = hf_hub_url(spec['id'], 'config.json', revision=spec['revision'])
        try:
            response = requests.head(url, headers={'Authorization': f'Bearer {token}'} if token else {}, timeout=30, allow_redirects=True)
            result['access_http_status'] = response.status_code
            result['access'] = 'approved_config_access' if response.ok else 'login_or_approval_required' if response.status_code in [401, 403] else 'unavailable'
        except requests.RequestException:
            result['access'] = 'network_check_failed'
    save_json(ROOT / 'artifacts/preflight.json', result)
    return result

def status_report():
    datasets = {}
    for dataset in ['ucf', 'scdd', 'scdd_confirmed_v1', 'elpv']:
        path = ROOT / 'data' / dataset / 'manifest.json'
        if not path.exists():
            continue
        manifest = read_json(path); rows = manifest['records']
        datasets[dataset] = {'images': len(rows), 'publication_eligible': manifest['publication_eligible'],
                             'partitions': {p: sum(r['partition'] == p for r in rows) for p in sorted({r['partition'] for r in rows})},
                             'excluded': len(manifest.get('excluded', [])), 'signature': manifest['signature']}
    runs = list((ROOT / 'artifacts/runs').glob('*/run.json'))
    dense_runs = [read_json(p) for p in (ROOT / 'artifacts/dense_runs').glob('*/run.json')]
    full_dense = [r for r in dense_runs if r['budget'] == 1.0 and r['epochs'] == 40]
    matrix = ROOT / 'artifacts/budget_matrix.json'
    has_dinov3 = (ROOT / 'artifacts/dinov3_smoke.json').exists()
    model_status = 'official Small weights loaded; real forward verified' if has_dinov3 else 'awaiting model access and local authentication'
    result = {'created_at': now(), 'datasets': datasets, 'trained_runs': len(runs),
              'full_dense_runs': len(full_dense), 'dense_smoke_runs': len(dense_runs) - len(full_dense),
              'budget_matrix_runs': len(read_json(matrix)['runs']) if matrix.exists() else 0,
              'dinov3_status': model_status,
              'dinov2': 'disabled by user', 'course_test': 'not imported; remains sealed',
              'implemented': ['pinned data acquisition', 'canonical masks and grouped splits', 'CLS/mean/concat features',
                              'Logistic/Linear SVM/RBF SVM', 'matched-supervision local and normal-reference heads',
                              'normal patch nearest-neighbor baseline', 'frozen held-out evaluation and native-mask metrics',
                              'Q/V LoRA and merged encoder export', 'full multi-depth DINOv3 decoder',
                              'COCO-pretrained DeepLabv3 / ResNet-50 full training',
                              'paired development bootstrap and source-driven English manuscript'],
              'pending': ['UCF real/synthetic provenance confirmation',
                          'full UCF cohort with preserved grouping', 'PVEL-AD download adapter',
                          'repeated dense-model seeds and label budgets', 'cross-domain and reverse-transfer confirmation',
                          'matched published semi-supervised reproduction', 'whole-dataset inference timing and deployment GPU peak-memory measurement']}
    if not has_dinov3:
        result['pending'].insert(0, 'real DINOv3 forward/benchmark after approval')
    save_json(ROOT / 'artifacts/status.json', result)
    rows_html = ''.join(f'<tr><td>{html.escape(d)}</td><td>{s["images"]}</td><td>{html.escape(str(s["partitions"]))}</td><td>{s["excluded"]}</td></tr>' for d, s in datasets.items())
    pending = ''.join(f'<li>{html.escape(item)}</li>' for item in result['pending'])
    status_text = f'{model_status}. Saved probe/local runs: {len(runs)}. Full dense architectures complete: {len(full_dense)}/5. DINOv2 is disabled.'
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>DINOv3 research readiness</title>
    <style>body{{font:17px/1.6 system-ui;max-width:1000px;margin:48px auto;padding:24px;color:#182b40;background:#f7fafc}}h1{{line-height:1.2}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ccd7df}}.status{{padding:18px;background:#fff1cf;border-radius:12px}}</style>
    <h1>Solar-cell crack research: local readiness</h1><p class="status">{html.escape(status_text)}</p>
    <p>Audited pilot cohorts. These are data counts, not experimental scores. UCF provenance remains unverified; current UCF studies must be described as exploratory.</p>
    <table><tr><th>Dataset</th><th>Usable images</th><th>Partitions</th><th>Excluded annotations</th></tr>{rows_html}</table>
    <p>ELPV uses the course development partition only. Its labels mean needs review, not crack ground truth. The 656 course test images remain sealed.</p>
    <h2>Pipeline and experiment scope</h2><p>Image probes and complete dense networks use source-only training. Normal references use only source training masks. Augmentation and acquisition families stay together. <a href="research_progress.html">Full architecture comparison and English manuscript</a> · <a href="initial_results.html">Initial source-validation results</a></p>
    <h2>Remaining work</h2><ul>{pending}</ul><p>Updated {html.escape(result['created_at'])}. All reported model scores must come from saved run artifacts.</p></html>'''
    (ROOT / 'artifacts/readiness_report.html').write_text(report)
    return result
