"""Native-annotation metrics that keep thin-defect and crack-negative failures visible."""
import numpy as np
from sklearn.metrics import average_precision_score

from .data import IGNORE


def native_crack_metrics(mask,probability,threshold=.5,normal_structure=None):
    mask=np.asarray(mask);score=np.asarray(probability,dtype=np.float64)
    if mask.ndim!=2 or score.shape!=mask.shape:raise ValueError('Native mask and probability shapes differ')
    if not np.isfinite(score).all() or score.min()<0 or score.max()>1:
        raise ValueError('Expected finite probabilities on a fixed [0,1] scale')
    valid=mask!=IGNORE
    if not valid.any():raise ValueError('No known annotation pixels')
    truth=(mask==1)&valid;prediction=(score>threshold)&valid;negative=valid&~truth
    intersection=int((truth&prediction).sum());union=int((truth|prediction).sum())
    total=int(truth.sum()+prediction.sum());fp=int((prediction&negative).sum())
    result={'known_pixels':int(valid.sum()),'crack_pixels':int(truth.sum()),'crack_positive':bool(truth.any()),
        'annotated_crack_fraction':float(truth.sum()/valid.sum()),'predicted_crack_pixels':int(prediction.sum()),
        'false_positive_pixels':fp,'known_noncrack_pixels':int(negative.sum()),
        'noncrack_pixel_false_positive_fraction':float(fp/negative.sum()) if negative.any() else None,
        'pixel_average_precision':float(average_precision_score(truth[valid],score[valid])) if truth.any() else None,
        'dice':float(2*intersection/total) if total else 1.,'iou':float(intersection/union) if union else 1.,
        'positive_image_dice':float(2*intersection/total) if truth.any() else None,
        'negative_image_false_positive_fraction':float(prediction.sum()/valid.sum()) if not truth.any() else None,
        'pixel_threshold':float(threshold),'probability_space':'native annotation pixels; no per-image display normalization'}
    if normal_structure is not None:
        region=np.asarray(normal_structure,dtype=bool)
        if region.shape!=mask.shape:raise ValueError('Normal structure mask shape differs')
        region=region&valid&~truth
        eligible=not truth.any()
        result['negative_image_normal_structure_pixels']=int(region.sum()) if eligible else None
        result['negative_image_normal_structure_false_positive_pixels']=int((region&prediction).sum()) if eligible else None
        result['negative_image_normal_structure_false_positive_fraction']=float((region&prediction).sum()/region.sum()) if eligible and region.any() else None
    return result


def pooled_native_metrics(items,threshold=.5):
    """items: dictionaries with canonical mask, native score, optionally raw normal_structure."""
    if not items:raise ValueError('No localization examples')
    summaries=[];truth=[];scores=[]
    for item in items:
        record=native_crack_metrics(item['mask'],item['score'],threshold,item.get('normal_structure'))
        summaries.append(record);valid=np.asarray(item['mask'])!=IGNORE
        truth.append(np.asarray(item['mask'])[valid]==1);scores.append(np.asarray(item['score'])[valid])
    positive=[r for r in summaries if r['crack_positive']];negative=[r for r in summaries if not r['crack_positive']]
    combined_truth=np.concatenate(truth)
    structural=[r for r in negative if r.get('negative_image_normal_structure_pixels')]
    return {'images':len(items),'positive_images':len(positive),'negative_images':len(negative),
        'pooled_pixel_average_precision':float(average_precision_score(combined_truth,np.concatenate(scores))) if combined_truth.any() else None,
        'mean_positive_image_dice':float(np.mean([r['dice'] for r in positive])) if positive else None,
        'mean_positive_image_iou':float(np.mean([r['iou'] for r in positive])) if positive else None,
        'mean_dice_all_images':float(np.mean([r['dice'] for r in summaries])),
        'mean_negative_image_false_positive_fraction':float(np.mean([r['negative_image_false_positive_fraction'] for r in negative])) if negative else None,
        'negative_images_with_structural_annotation':len(structural),
        'pooled_negative_structure_false_positive_fraction':sum(r['negative_image_normal_structure_false_positive_pixels'] for r in structural)/sum(r['negative_image_normal_structure_pixels'] for r in structural) if structural else None,
        'pixel_threshold':float(threshold),'per_image':summaries,
        'limits':['All-image Dice includes empty-mask true negatives; positive-image Dice is reported separately.',
                  'Pooled pixel AP is a point estimate; pixels are not independent bootstrap units.',
                  'Normal-structure FPR uses annotated structure pixels of crack-negative images only.']}


def box_point_localization(score_map,normalized_boxes):
    """Top-response location in normalized author boxes; a coarse localization diagnostic, not detection AP."""
    score=np.asarray(score_map,dtype=np.float64);boxes=np.asarray(normalized_boxes,dtype=np.float64)
    if score.ndim!=2 or not np.isfinite(score).all():raise ValueError('Expected a finite native 2D response map')
    if boxes.size==0:raise ValueError('Point localization is defined only for images with target-class boxes')
    if boxes.ndim!=2 or boxes.shape[1]!=4 or not np.isfinite(boxes).all() or (boxes<0).any() or (boxes>1).any():
        raise ValueError('Boxes must be finite normalized xyxy coordinates')
    if (boxes[:,2]<=boxes[:,0]).any() or (boxes[:,3]<=boxes[:,1]).any():raise ValueError('Empty or inverted box')
    height,width=score.shape;row,column=np.unravel_index(int(score.argmax()),score.shape)
    x,y=(column+.5)/width,(row+.5)/height
    inside=bool(((boxes[:,0]<=x)&(x<=boxes[:,2])&(boxes[:,1]<=y)&(y<=boxes[:,3])).any())
    yy,xx=np.meshgrid((np.arange(height)+.5)/height,(np.arange(width)+.5)/width,indexing='ij')
    covered=np.zeros(score.shape,dtype=bool)
    for xmin,ymin,xmax,ymax in boxes:covered|=(xx>=xmin)&(xx<=xmax)&(yy>=ymin)&(yy<=ymax)
    return {'maximum_point_in_target_box':inside,'maximum_point_normalized_xy':[float(x),float(y)],
            'target_box_count':len(boxes),'box_union_pixel_center_fraction':float(covered.mean()),
            'tie_policy':'first row-major pixel on exact response ties',
            'scope':'coarse box localization; not pixel segmentation quality or object-detection mAP'}
