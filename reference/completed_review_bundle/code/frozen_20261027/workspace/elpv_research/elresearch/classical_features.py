"""Label-independent HOG descriptors and explicitly frozen DINOv3 cache reuse."""
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch.nn import functional as F
from skimage.feature import hog
from skimage.transform import resize

from .common import ROOT,digest,read_json,sha256
from .data import load_image


HOG_SETTINGS={'size':128,'orientations':9,'pixels_per_cell':16,'cells_per_block':2,
              'block_norm':'L2-Hys','transform_sqrt':False,'statistics_dimensions':42,'hog_dimensions':1764}


def image_statistics(gray):
    gray=np.asarray(gray,dtype=np.float64)/255.
    dy,dx=np.gradient(gray);gradient=np.hypot(dx,dy)
    values=[gray.mean(),gray.std(),*np.quantile(gray,[.1,.5,.9]),
            (gray<.15).mean(),(gray<.30).mean(),gradient.mean(),gradient.std(),np.quantile(gradient,.9)]
    for row in np.array_split(gray,4,axis=0):
        for block in np.array_split(row,4,axis=1):values.extend([block.mean(),block.std()])
    return np.asarray(values)


def hog_statistics(gray):
    gray=np.asarray(gray,dtype=np.uint8)
    small=resize(gray,(128,128),anti_aliasing=True)
    gradient=hog(small,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),
                 block_norm='L2-Hys',transform_sqrt=False)
    result=np.r_[image_statistics(gray),gradient].astype(np.float32)
    if result.shape!=(1806,) or not np.isfinite(result).all():raise ValueError('Invalid HOG/statistics descriptor')
    return result


class FeatureBank:
    def __init__(self,rows,values,identity):
        self.ids=[r['id'] for r in rows];self.index={name:i for i,name in enumerate(self.ids)}
        self.image_hashes={r['id']:r['image_sha256'] for r in rows}
        self.values=np.asarray(values,dtype=np.float32);self.identity=identity
        if len(self.index)!=len(rows):raise ValueError('Feature bank has duplicate identities')
        if self.values.ndim!=3 or self.values.shape[:2]!=(len(rows),8) or not np.isfinite(self.values).all():
            raise ValueError('Expected finite [images, eight D4 views, dimensions] features')
        self.dimensions=self.values.shape[-1]
        self.feature_sha256=hashlib.sha256(self.values.tobytes()).hexdigest()

    def get(self,rows,augmented=False):
        if any(self.image_hashes[r['id']]!=r['image_sha256'] for r in rows):raise RuntimeError('Feature/image identity changed')
        value=self.values[[self.index[r['id']] for r in rows]]
        return value.reshape(-1,self.dimensions).astype(np.float64) if augmented else value[:,0].astype(np.float64)


def hog_bank(rows):
    if any(r['partition'] not in {'train','validation','target_development'} for r in rows):
        raise PermissionError('This extractor is restricted to predeclared development images')
    root=ROOT/'cache/publication_hog_d4';root.mkdir(parents=True,exist_ok=True);values=[]
    identity={'kind':'hog_statistics','settings':HOG_SETTINGS,'feature_code_sha256':sha256(__file__),
              'preprocessing':'native grayscale statistics; full-image resize to 128x128 for HOG; no learned feature statistics here'}
    for index,row in enumerate(rows):
        key=digest({'id':row['id'],'image_sha256':row['image_sha256'],'descriptor':identity})
        path=root/f'{key}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as stored:value=stored['features'].copy()
        else:
            gray=np.asarray(load_image(row),dtype=np.uint8);descriptors=[]
            for code in range(8):
                view=np.rot90(gray,code%4)
                if code>=4:view=np.fliplr(view)
                descriptors.append(hog_statistics(view))
            value=np.stack(descriptors);temporary=path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,features=value);temporary.replace(path)
        if value.shape!=(8,1806):raise RuntimeError('HOG cache shape differs')
        values.append(value)
        if (index+1)%200==0:print('HOG D4 features',index+1,'/',len(rows),flush=True)
    return FeatureBank(rows,np.stack(values),identity)


def frozen_dino_bank(parent_protocol,parent_frozen_run,rows):
    if parent_frozen_run['model']['encoder']!='dinov3_small' or parent_frozen_run['model']['adaptation']!='frozen':
        raise PermissionError('Only the unchanged pretrained encoder cache may be shared across source/target roles')
    identity=parent_frozen_run['encoder']
    key=digest({'encoder':identity,'preprocessing_sha256':parent_protocol['scientific_hashes']['encoders.py'],
        'model_code_sha256':parent_protocol['scientific_hashes']['publication_models.py'],
        'images':[(r['id'],r['image_sha256']) for r in rows],
        'views':'rot90 k=0..3, optionally horizontal flip after rotation; float32 CLS'})
    path=ROOT/'cache/publication_cls_d4'/f'{key}.npz'
    if not path.exists():raise FileNotFoundError('Registered frozen DINOv3 cache is not ready: '+str(path))
    with np.load(path,allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['ids'],[r['id'] for r in rows]);raw=stored['features'].copy()
    if raw.shape!=(len(rows),8,384):raise RuntimeError('Unexpected DINOv3-S feature cache dimensions')
    # Exactly the per-image non-affine LayerNorm used by the neural linear head.
    torch.set_num_threads(2)
    normalized=F.layer_norm(torch.from_numpy(raw),(384,)).numpy()
    feature_identity={'kind':'dinov3_cls','encoder':identity,'raw_cache_path':str(path.relative_to(ROOT)),
        'raw_cache_sha256':sha256(path),'normalization':'per-image non-affine LayerNorm, eps=1e-5; no fitted source/target scaler',
        'feature_code_sha256':sha256(__file__)}
    return FeatureBank(rows,normalized,feature_identity)
