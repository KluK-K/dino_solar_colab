"""Native confirmation annotations; explicit unknown/ignore policies, no target fitting."""
import numpy as np
from PIL import Image

from .common import ROOT, sha256
from .data import IGNORE, canonical_mask
from .pvelad_data import read_annotation


def native_annotation(row):
    if row.get('dataset')=='elpv' or row.get('partition') not in {'test','external'}:
        raise PermissionError('This loader is only for registry-authorized research confirmation')
    if row['target'] not in [0,1]:raise ValueError('Unknown crack label is not a negative')
    if row['dataset'] in {'scdd','ucf'}:
        path=ROOT/row['mask_path']
        if sha256(path)!=row['mask_sha256']:raise RuntimeError('Confirmation mask changed')
        with Image.open(path) as im:raw=np.asarray(im).copy()
        if row['dataset']=='ucf':
            if raw.ndim!=2 or not set(np.unique(raw))<={0,1,2,3,4,128}:raise ValueError('Unexpected UCF annotation indices')
            mask=np.full(raw.shape,IGNORE,dtype=np.uint8)
            mask[raw==0]=0;mask[raw==1]=1;mask[np.isin(raw,[2,3,4])]=2
        else:mask=canonical_mask(raw,'scdd')[0]
        if int((mask==1).any())!=row['target']:raise ValueError('Image crack label and native mask disagree')
        return {'kind':'pixel_mask','mask':mask,'normal_structure':np.isin(raw,[4,9]) if row['dataset']=='scdd' else None}
    if row['dataset']=='pvelad':
        if row['role']=='known_annotated_crack_positive' and row['target']==1:
            path=ROOT/row['annotation_path']
            if sha256(path)!=row['annotation_sha256']:raise RuntimeError('PVEL annotation changed')
            annotation=read_annotation(path)
            boxes=[b['normalized_xyxy'] for b in annotation['boxes'] if b['class'] in {'crack','star_crack'}]
            if not boxes or annotation['boxes']!=row['boxes']:raise ValueError('PVEL frozen box identity differs')
            return {'kind':'boxes','boxes':boxes,'shape':[annotation['height'],annotation['width']]}
        if row['target']==0 and row['role']=='author_normal_negative':
            return {'kind':'author_normal','mask':None,'limits':'Directory label; no independent exhaustive pixel annotation'}
        raise PermissionError('Other-defect PVEL crack negatives are unknown; missing XML is not a negative label')
    raise PermissionError('Unregistered dataset annotation semantics')
