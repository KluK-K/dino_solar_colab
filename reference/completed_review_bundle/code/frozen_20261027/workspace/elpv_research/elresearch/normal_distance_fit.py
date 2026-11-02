"""Source-calibrated normal-memory baseline, without classifier/encoder optimization."""
import gc
import numpy as np
import torch

from .common import ROOT,read_json,save_json,sha256,now,choose_device
from .data import IGNORE
from .image_classification import assert_source_partitions,atomic_torch_save
from .metrics import classification,recall_threshold
from .publication_train import budget_rows,calibration,probability_metrics
from .publication_dense_data import prepare_dense_rows,augmented_batch
from .publication_dense_train import selection_predictions,native_evaluation
from .mechanistic_models import build_mechanistic_system
from .mechanistic_train import scientific_hashes


def fit_normal_distance(settings,spec,source_train,source_validation,seed,budget,folder):
    if scientific_hashes()!=settings['scientific_hashes']:raise RuntimeError('Normal-distance scientific implementation changed')
    assert_source_partitions(source_train,source_validation)
    if (folder/'run.json').exists():
        result=read_json(folder/'run.json')
        if result['checkpoint_sha256']!=sha256(folder/'best.pt') or result['study_id']!=settings['study_id']:
            raise RuntimeError('Completed normal-distance control changed')
        return result
    folder.mkdir(parents=True,exist_ok=True);selected=budget_rows(source_train,budget,seed)
    device=choose_device(settings['device']);torch.set_num_threads(4)
    model,identity,bank=build_mechanistic_system(spec,seed,settings['input_size'],device,selected)
    bank.write_recipe(folder);validation=prepare_dense_rows(source_validation,settings['input_size'])
    normal_distances=[]
    with torch.inference_mode():
        for start in range(0,len(validation),settings['batch_size']):
            items=validation[start:start+settings['batch_size']]
            b=augmented_batch(items,[0]*len(items),device)
            raw=model(b['images'],b['valid'],b['geometries'],b['groups'])['raw_distance'].cpu().numpy()
            for item,value in zip(items,raw):
                indices=item['selection_indices'];target=item['target'].numpy().ravel()[indices]
                keep=(target!=IGNORE)&(target!=1)
                normal_distances.append(value.ravel()[indices][keep])
    values=np.concatenate(normal_distances)
    if not len(values) or not np.isfinite(values).all():raise ValueError('No source-validation noncrack responses for a fixed normal-FPR cutoff')
    cutoff=float(np.quantile(values,.99))
    state={'pixel_cutoff':torch.tensor(cutoff)};model.load_calibration_state(state)
    atomic_torch_save(state,folder/'best.pt')
    calibrated_spec={**spec,'pixel_distance_cutoff':cutoff}
    logits,local=native_evaluation(model,validation,device,settings['batch_size'],folder)
    before,scores,_=selection_predictions(model,validation,device,settings['batch_size'])
    del model;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    restored,again_identity,_=build_mechanistic_system(calibrated_spec,seed,settings['input_size'],device,bank=bank)
    restored.load_calibration_state(torch.load(folder/'best.pt',map_location='cpu',weights_only=True))
    again,again_scores,_=selection_predictions(restored,validation,device,settings['batch_size'])
    if again_identity!=identity:raise RuntimeError('Normal-memory encoder changed on reload')
    np.testing.assert_allclose(before,again,atol=2e-5,rtol=2e-5);np.testing.assert_allclose(scores,again_scores,atol=2e-5,rtol=2e-5)
    y=np.array([r['target'] for r in source_validation]);threshold=recall_threshold(y,logits,settings['target_source_recall'])
    temperature=calibration(y,logits)
    result={'completed_at':now(),'study_id':settings['study_id'],'model':calibrated_spec,'seed':seed,
        'training_budget_fraction':budget,'training_ids':[r['id'] for r in selected],
        'validation_ids':[r['id'] for r in source_validation],'encoder':identity,'scientific_hashes':settings['scientific_hashes'],
        'training_images':len(selected),'validation_images':len(source_validation),
        'total_pixel_annotated_images_including_validation':len(selected)+len(source_validation),
        'trainable_parameters':0,'total_parameters':sum(p.numel() for p in restored.parameters()),
        'source_logit_threshold':threshold,'source_temperature':temperature,
        'validation_metrics':classification(y,logits,threshold),'validation_localization':local,
        'validation_calibration_temperature_scaled':probability_metrics(y,logits,temperature),
        'checkpoint_sha256':sha256(folder/'best.pt'),
        'checkpoint_reload':{'fresh_model':True,'logit_max_abs_error':float(np.max(abs(before-again))),
            'sampled_pixel_probability_max_abs_error':float(np.max(abs(scores-again_scores)))},
        'pixel_operating_point':{'raw_cosine_distance_cutoff':cutoff,'source_noncrack_quantile':.99,
            'selection_sample_pixels':len(values),'response_scale':10.,
            'limits':'A source noncrack-FPR decision mapping, not a calibrated crack posterior. Pixel AP is primary; its cutoff convention differs from learned segmentation posterior0.5.'},
        'memory':'At most256 source images; up to64 evenly indexed normal patches per image and D4 view. Exact cosine NN within this capped memory, chunk2048. Entire query capture family excluded.',
        'supervision':'Selected source masks identify eligible normal patches; source validation known noncrack pixels set the 99% response cutoff; image labels set source recall/temperature. Not an annotation-free baseline.',
        'scope':'Normal-memory method control inspired by published patch-memory approaches; not an exact AnomalyDINO/SuperADD reproduction or physical crack posterior',
        'seed_interpretation':'Only training image subsets vary. Full-budget deterministic memory saved once.'}
    np.savez_compressed(folder/'validation_predictions.npz',ids=np.array([r['id'] for r in source_validation]),
        y=y,logits=logits,groups=np.array([r['split_group'] for r in source_validation]))
    save_json(folder/'run.json',result);del restored,bank,validation;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    return result
