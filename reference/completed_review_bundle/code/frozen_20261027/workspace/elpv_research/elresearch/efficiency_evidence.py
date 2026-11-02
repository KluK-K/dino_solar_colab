"""Verify full source-cohort coverage rather than accepting a timing completion flag."""
from pathlib import Path
import math

import numpy as np

from .common import ROOT,read_json,sha256
from .confirmation_registry import bind_model,check_file


def verify_completed_benchmark(folder,require_complete_status=True):
    folder=Path(folder);protocol_path=folder/'protocol.json';protocol=read_json(protocol_path)
    if require_complete_status:
        status=read_json(folder/'status.json')
        if status['state']!='complete' or status['completed_jobs']!=protocol['planned_jobs']:
            raise RuntimeError('Formal efficiency evidence is incomplete')
    source=read_json(folder/'model_bindings.json');jobs=source['jobs']
    if source['protocol_sha256']!=sha256(protocol_path) or len(jobs)!=protocol['planned_jobs']:
        raise RuntimeError('Efficiency model/protocol coverage differs')
    expected={f"{s['family']}_seed{s['seed']}":s for s in protocol['models']}
    if len(expected)!=protocol['planned_jobs'] or {j['id'] for j in jobs}!=set(expected):
        raise RuntimeError('Efficiency omitted or repeated a registered model/seed/view')
    checked=[]
    for job in jobs:
        job_id=job['id'];binding=job['binding'];spec=expected[job_id]
        if (binding['kind']!=spec['kind'] or binding['family']!=spec['family'] or binding['seed']!=spec['seed']
            or binding['source_view']!=spec['source_view'] or (ROOT/spec['run_folder']).resolve()!=check_file(binding['run']).parent):
            raise RuntimeError('Efficiency measured a different source model/view')
        canonical=bind_model(binding['kind'],binding['family'],binding['seed'],check_file(binding['run']).parent,
            check_file(binding['source_protocol']),binding['source_view'])
        if canonical!=binding:raise RuntimeError('Efficiency source checkpoint/operating point changed')
        result_path=folder/'jobs'/job_id/'result.json';result=read_json(result_path)
        if (result['model_id']!=job_id or result['study_id']!=protocol['study_id']
            or result['protocol_sha256']!=sha256(protocol_path)
            or result['model_bindings_sha256']!=sha256(folder/'model_bindings.json')
            or result['binding']!=binding or result['prediction_equivalence_verified'] is not True):
            raise RuntimeError('Efficiency result identity/proof differs')
        if set(result['cohorts'])!=set(protocol['cohorts']) or set(result['numerical_signature_files'])!=set(protocol['cohorts']):
            raise RuntimeError('Efficiency omitted a source-input cohort or prediction signature')
        for cohort,rows in protocol['cohorts'].items():
            with np.load(check_file(result['numerical_signature_files'][cohort]),allow_pickle=False) as f:
                np.testing.assert_array_equal(f['ids'],[r['id'] for r in rows])
                if f['logits'].shape!=(len(rows),) or not np.isfinite(f['logits']).all():raise RuntimeError('Invalid benchmark image signatures')
                offsets=f['native_sample_offsets'];values=f['native_sample_values']
                if len(offsets)!=len(rows)+1 or offsets[0]!=0 or offsets[-1]!=len(values) or (np.diff(offsets)<0).any():
                    raise RuntimeError('Benchmark native signatures do not cover the input images')
                if binding['kind']=='dense' and not (np.diff(offsets)>0).all():raise RuntimeError('Dense timing silently lost native localization outputs')
                if not np.isfinite(values).all() or (len(values) and (values.min()<0 or values.max()>1)):
                    raise RuntimeError('Benchmark native signatures are not raw probabilities')
            cases=result['cohorts'][cohort]
            if len(cases)!=len(protocol['batch_sizes']) or {c['batch_size'] for c in cases}!=set(protocol['batch_sizes']):
                raise RuntimeError('Benchmark omitted a registered batch size')
            for case in cases:
                repeats=case['repeated_uninstrumented_passes'];raw=case['raw_batch_timings'];batch_size=case['batch_size']
                if len(repeats)!=protocol['repetitions'] or len(raw)!=len(repeats):raise RuntimeError('Benchmark timing repetitions incomplete')
                for summary,batches in zip(repeats,raw):
                    if (summary['images']!=len(rows) or summary['batches']!=len(batches)
                        or any(b['images']<1 or b['images']>batch_size for b in batches)
                        or sum(b['images'] for b in batches)!=len(rows)):
                        raise RuntimeError('Benchmark timing did not process the complete cohort')
                    seconds=sum(b['seconds'] for b in batches)
                    if not math.isfinite(seconds) or seconds<=0 or not math.isclose(seconds,summary['timed_pipeline_seconds'],rel_tol=1e-10):
                        raise RuntimeError('Benchmark timer denominator differs')
                    if not math.isclose(len(rows)/seconds,summary['pipeline_images_per_second'],rel_tol=1e-10):
                        raise RuntimeError('Benchmark throughput uses the wrong image count')
                    for batch in batches:
                        keys=set(batch['stages'])
                        if any(k.startswith('component_') for k in keys):raise RuntimeError('Synchronized component times replaced uninstrumented latency')
                        if not {'read_checksum_decode','preprocessing','model_forward_and_transfers','decision_and_native_reconstruction'}<=keys:
                            raise RuntimeError('Benchmark excludes part of the full inference pipeline')
                if case['component_pass']['images']!=len(rows) or case['memory']['samples']<1:
                    raise RuntimeError('Component/memory profile did not cover its source cohort')
                proofs=case['prediction_equivalence']
                if len(proofs)!=protocol['repetitions']+1:raise RuntimeError('Cross-pass/component prediction equivalence missing')
                for proof in proofs:
                    for key in ['maximum_logit_scaled_tolerance_ratio','maximum_pixel_scaled_tolerance_ratio']:
                        if not 0<=proof[key]<=1.000001:raise RuntimeError('Timing instrumentation/batching changed numerical predictions')
        checked.append({'id':job_id,'result_path':str(result_path.relative_to(ROOT)),'result_sha256':sha256(result_path)})
    return {'verified_jobs':len(checked),'planned_jobs':protocol['planned_jobs'],'results':checked,
        'scope':'Actual bound source models, both complete source cohorts/batch sizes, every repetition and numerical signature checked. This is inference-cost evidence, not accuracy or independent confirmation.'}
