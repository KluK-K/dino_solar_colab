"""Exact-registry confirmation, resumable by model/cohort, bounded in image memory."""
import fcntl
import gc
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .common import ROOT, read_json, save_json, digest, now, sha256, choose_device
from .confirmation_registry import authorize_rows, bind_file, check_file
from .confirmation_predict import FrozenScorer
from .confirmation_data import native_annotation
from .localization_metrics import native_crack_metrics, box_point_localization
from .metrics import classification
from .publication_train import probability_metrics
from .streaming_inference import _stream_image_scores


class NativeAccumulator:
    """Keep per-image summaries; exact pooled AP uses disk arrays, not all image maps."""
    def __init__(self,folder,fixed_ids=()):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.rows=[];self.boxes=[];self.normal=[];self.files=[];self.fixed_ids=set(fixed_ids)

    def consume(self,row,logit,probability):
        annotation=native_annotation(row)
        if row['id'] in self.fixed_ids:
            destination=self.folder.parent/'fixed_native_examples';destination.mkdir(exist_ok=True)
            np.savez_compressed(destination/(digest(row['id'])+'.npz'),id=np.array(row['id']),
                                crack_probability=probability.astype(np.float16))
        if annotation['kind']=='pixel_mask':
            mask=annotation['mask'];valid=mask!=255
            summary=native_crack_metrics(mask,probability,.5,annotation['normal_structure'])
            self.rows.append({'id':row['id'],'group':row['split_group'],
                              'acquisition_source':row.get('acquisition_source'),**summary})
            # Only small SCDD/UCF pixel cohorts reach this path. PVEL boxes/good
            # never create hundreds of GB of dense map caches.
            path=self.folder/(digest(row['id'])+'.npz')
            np.savez_compressed(path,truth=(mask[valid]==1),score=probability[valid].astype(np.float32))
            self.files.append(path)
        elif annotation['kind']=='boxes':
            if list(probability.shape)!=annotation['shape']:raise ValueError('PVEL native-map shape differs from XML')
            self.boxes.append({'id':row['id'],**box_point_localization(probability,annotation['boxes'])})
        else:
            self.normal.append({'id':row['id'],'predicted_crack_fraction_at_0_5':float((probability>.5).mean()),
                                'interpretation':'Author-normal directory image; no pixel ground-truth annotation'})

    def finish(self):
        result={'pixel_threshold':.5,'pixel_images':len(self.rows),'box_positive_images':len(self.boxes),
                'author_normal_images':len(self.normal),'native_pixel_per_image':self.rows,
                'box_per_image':self.boxes,'normal_response_per_image':self.normal,
                'limits':['Fixed probability scale; no per-image activation normalization.',
                          'Pixel AP is full native annotation; pixels are not independent replicates.',
                          'PVEL box-point coverage is not pixel segmentation AP or detection mAP.']}
        if self.files:
            truth=[];score=[]
            for path in self.files:
                with np.load(path,allow_pickle=False) as f:truth.append(f['truth'].copy());score.append(f['score'].copy())
            result['pooled_native_pixel_AP']=float(average_precision_score(np.concatenate(truth),np.concatenate(score)))
            positive=[r for r in self.rows if r['crack_positive']]
            negative=[r for r in self.rows if not r['crack_positive']]
            structure=[r for r in negative if r.get('negative_image_normal_structure_pixels')]
            result['mean_positive_image_dice']=float(np.mean([r['dice'] for r in positive])) if positive else None
            result['mean_positive_image_iou']=float(np.mean([r['iou'] for r in positive])) if positive else None
            result['mean_negative_image_pixel_FPR']=float(np.mean([r['negative_image_false_positive_fraction'] for r in negative])) if negative else None
            result['pooled_negative_structure_pixel_FPR']=(sum(r['negative_image_normal_structure_false_positive_pixels'] for r in structure)/sum(r['negative_image_normal_structure_pixels'] for r in structure)) if structure else None
        if self.boxes:
            result['box_point_hit_rate']=float(np.mean([r['maximum_point_in_target_box'] for r in self.boxes]))
            result['mean_box_coverage_fraction']=float(np.mean([r['box_union_pixel_center_fraction'] for r in self.boxes]))
        return result


def evaluate(folder,model_id,cohort_name,device='auto'):
    folder=Path(folder)
    registry,binding,cohort,rows=authorize_rows(folder,model_id,cohort_name)
    if choose_device(device).type!=registry['analysis_plan']['inference_device']:
        raise RuntimeError('Requested backend differs from frozen confirmation inference device')
    # A single model/cohort lock prevents duplicate jobs and competing writes.
    job=folder/'results'/model_id/cohort_name;job.mkdir(parents=True,exist_ok=True)
    with (job/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (job/'result.json').exists():
            result=read_json(job/'result.json')
            if result['registry_id']!=registry['registry_id'] or sha256(job/'predictions.npz')!=result['predictions_sha256']:
                raise RuntimeError('Completed confirmation output changed')
            return result
        save_json(job/'started.json',{'at':now(),'registry_id':registry['registry_id'],'model':model_id,
                                     'cohort':cohort_name,'pid':os.getpid(),'partial_failure_counts_as_observed':True})
        scorer=FrozenScorer(binding,device)
        fixed_ids=registry['analysis_plan'].get('fixed_localization_examples',{}).get(cohort_name,[])
        if not set(fixed_ids)<={r['id'] for r in rows}:raise ValueError('Predeclared localization examples must belong to this cohort')
        local=NativeAccumulator(job/'native_pixel_cache',fixed_ids) if binding['kind']=='dense' else None
        output=_stream_image_scores(rows,scorer,scorer.size,scorer.batch_size,local.consume if local else None)
        ids=[r['id'] for r in rows]
        np.testing.assert_array_equal(output['ids'],ids)
        y=np.array([r['target'] for r in rows]);logits=output['logits']
        destination=job/'predictions.npz';temporary=job/'predictions.tmp.npz'
        np.savez_compressed(temporary,ids=output['ids'],y=y,logits=logits,
                            groups=np.array([r.get('split_group') or '' for r in rows]))
        temporary.replace(destination)
        metrics=classification(y,logits,binding['source_threshold'])
        result={'at':now(),'registry_id':registry['registry_id'],'model_id':model_id,'cohort':cohort_name,
                'task_scope':cohort['task_scope'],'source_frozen_threshold':binding['source_threshold'],
                'source_frozen_temperature':binding['source_temperature'],'metrics':metrics,
                'raw_probability_metrics':probability_metrics(y,logits),
                'source_temperature_probability_metrics':probability_metrics(y,logits,binding['source_temperature']),
                'localization':local.finish() if local else None,'predictions_sha256':sha256(destination),
                'streaming':{k:v for k,v in output.items() if k not in {'ids','logits'}},
                'uncertainty_unit':cohort['uncertainty_unit'],'threshold_fitted_on_target':False,
                'limits':['Operating point and model selection frozen before confirmation access.',
                          'UCF individual real/simulated provenance is conditional on author release policy.',
                          'PVEL is a restricted known-crack/author-normal cohort, not specificity against other defects or the official test benchmark.']}
        source=read_json(check_file(registry['source_identity_index']))
        by_id={r['id']:r for r in source['records']}
        fit_ids=set(binding['source_train_ids']+binding['source_validation_ids'])
        unlabelled_ids=set(binding.get('source_unlabelled_ids',[]))
        selection_ids=set(binding['hyperparameter_selection_exposure_ids'])
        fit_datasets=sorted({by_id[i]['dataset'] for i in fit_ids})
        selection_datasets=sorted({by_id[i]['dataset'] for i in selection_ids})
        unlabelled_datasets=sorted({by_id[i]['dataset'] for i in unlabelled_ids})
        target_datasets={r['dataset'] for r in rows}
        result['development_exposure']={'fit_or_checkpoint_validation_datasets':fit_datasets,
            'hyperparameter_calibration_datasets':selection_datasets,
            'unlabelled_source_images':len(unlabelled_ids),'unlabelled_source_datasets':unlabelled_datasets,
            'target_dataset_used_for_unlabelled_fitting':bool(target_datasets&set(unlabelled_datasets)),
            'fit_and_validation_images':len(fit_ids),'additional_hyperparameter_calibration_images':len(selection_ids-fit_ids),
            'target_dataset_used_for_fitting_or_checkpoint_validation':bool(target_datasets&set(fit_datasets)),
            'target_dataset_used_for_hyperparameter_selection':bool(target_datasets&set(selection_datasets)),
            'interpretation':'Independent confirmation identities do not imply a never-observed dataset. Model-specific training and selection exposure is reported explicitly.'}
        source_metadata={r['id']:r for binding_ in source['source_manifests']
                         for r in read_json(check_file(binding_))['records']}
        def source_key(row):
            return (row['dataset'],str(row.get('acquisition_source') or row['dataset']).casefold())
        fit_sources={source_key(source_metadata[i]) for i in fit_ids}
        selection_sources={source_key(source_metadata[i]) for i in selection_ids}
        unlabelled_sources={source_key(source_metadata[i]) for i in unlabelled_ids}
        result['predeclared_acquisition_source_strata']={}
        for key in sorted({source_key(r) for r in rows}):
            keep=np.array([source_key(r)==key for r in rows])
            result['predeclared_acquisition_source_strata']['/'.join(key)]={
                'metrics':classification(y[keep],logits[keep],binding['source_threshold']),
                'source_used_for_fitting_or_checkpoint_validation':key in fit_sources,
                'source_used_for_hyperparameter_selection':key in selection_sources,
                'source_used_for_unlabelled_fitting':key in unlabelled_sources,
                'interpretation':'Same frozen model and operating point; acquisition labels do not prove physical-module identity.'}
        if cohort_name.startswith('pvelad'):
            result['primary_restricted_endpoints']={'known_crack_recall':metrics['recall'],
                'author_normal_FPR':metrics['fp']/(metrics['fp']+metrics['tn']),
                'known_crack_images':metrics['positive'],'author_normal_images':metrics['n']-metrics['positive']}
            result['secondary_composition_dependent_metrics']=['AUROC','average_precision','macro_F1','review_rate','calibration']
            result['physical_acquisition_group_CI']=None
            line=np.array([r.get('target_line_crack') is not None for r in rows])
            result['line_crack_only_sensitivity']={'excluded_star_only_images':int((~line).sum()),
                'metrics':classification(y[line],logits[line],binding['source_threshold']),
                'interpretation':'Same fixed model/threshold; star-only images excluded, author-normal negatives retained'}
        save_json(job/'result.json',result)
        del scorer,local;gc.collect()
        if torch.backends.mps.is_available():torch.mps.empty_cache()
        return result
