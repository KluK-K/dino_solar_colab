"""Development folds cannot include either confirmation cohort."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .common import ROOT,digest,read_json,sha256
from .image_classification import assert_source_partitions
from .publication_train import budget_rows


DATASET='scdd_publication_v1'


def partition_manifest():
    path=ROOT/'data'/DATASET/'manifest.json';manifest=read_json(path)
    development=[r for r in manifest['records'] if r['partition'] in {'train','validation'}]
    confirmation=[r for r in manifest['records'] if r['partition']=='test']
    if len(development)!=471 or len(confirmation)!=117:
        raise ValueError('This protocol is specific to the audited 471/117 cohort')
    if {r['split_group'] for r in development}&{r['split_group'] for r in confirmation}:
        raise ValueError('Confirmation acquisition-family overlap')
    if {r['d4_pixel_sha256'] for r in development}&{r['d4_pixel_sha256'] for r in confirmation}:
        raise ValueError('Confirmation D4 duplicate leakage')
    folds=[]
    train=[r for r in development if r['partition']=='train']
    validation=[r for r in development if r['partition']=='validation']
    assert_source_partitions(train,validation)
    folds.append({'name':'internal','source_train_ids':[r['id'] for r in train],
                  'source_validation_ids':[r['id'] for r in validation],
                  'target_development_ids':[],'budgets':[.05,.1,.2,1.],
                  'role':'within-source development; original train and validation retained'})
    for held_out in ['ARTS','CFVS','CSIR']:
        source=sorted([r for r in development if r['acquisition_source']!=held_out],key=lambda r:r['id'])
        target=sorted([r for r in development if r['acquisition_source']==held_out],key=lambda r:r['id'])
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261017)
        train_index,val_index=next(cv.split(np.zeros(len(source)),[r['target'] for r in source],
                                           [r['split_group'] for r in source]))
        train=[{**source[i],'partition':'train'} for i in train_index]
        validation=[{**source[i],'partition':'validation'} for i in val_index]
        assert_source_partitions(train,validation)
        if {r['target'] for r in target}!={0,1}:raise ValueError('Held-out domain lacks a class')
        for rowset in [train,validation]:
            if {r['split_group'] for r in rowset}&{r['split_group'] for r in target}:
                raise ValueError('Source/target acquisition-family overlap')
            if {r['d4_pixel_sha256'] for r in rowset}&{r['d4_pixel_sha256'] for r in target}:
                raise ValueError('Source/target D4 duplicate leakage')
        folds.append({'name':'holdout_'+held_out,'held_out_acquisition_source':held_out,
                      'source_train_ids':[r['id'] for r in train],
                      'source_validation_ids':[r['id'] for r in validation],
                      'target_development_ids':[r['id'] for r in target],'budgets':[.2,1.],
                      'role':'leave-one-acquisition-source-out DEVELOPMENT; never final confirmation'})
    details=[]
    by_id={r['id']:r for r in development}
    for fold in folds:
        train=[by_id[i] for i in fold['source_train_ids']]
        row={**fold,'counts':{key:{'images':len(fold[key]),
                        'positive':sum(by_id[i]['target'] for i in fold[key]),
                        'capture_groups':len({by_id[i]['split_group'] for i in fold[key]})}
                        for key in ['source_train_ids','source_validation_ids','target_development_ids']},
             'annotation_budget_counts':[]}
        for seed in [11,22,33]:
            for budget in fold['budgets']:
                selected=budget_rows(train,budget,seed)
                row['annotation_budget_counts'].append({'seed':seed,'training_fraction':budget,
                    'training_images':len(selected),'training_positives':sum(r['target'] for r in selected),
                    'training_ids_sha256':digest([r['id'] for r in selected]),
                    'validation_images':len(fold['source_validation_ids']),
                    'total_labeled_images_including_validation':len(selected)+len(fold['source_validation_ids'])})
        details.append(row)
    return {'dataset':DATASET,'manifest_sha256':sha256(path),'manifest_signature':manifest['signature'],
            'development_images':len(development),'confirmation_images_remaining_sealed':len(confirmation),
            'confirmation_ids_sha256':digest([r['id'] for r in confirmation]),
            'split_code_sha256':sha256(__file__),'folds':details,
            'limits':['Sources name laboratories or collection channels; physical module IDs are unconfirmed.',
                      'LOSO development labels will be observed in other folds; it is not a blind test.',
                      'Budget percentages apply to training image labels; validation labels are additionally counted.']}


def fold_records(split_manifest,fold_name):
    path=ROOT/'data'/split_manifest['dataset']/'manifest.json'
    if sha256(path)!=split_manifest['manifest_sha256']:raise RuntimeError('Registered cohort changed')
    by_id={r['id']:r for r in read_json(path)['records'] if r['partition']!='test'}
    fold=next(f for f in split_manifest['folds'] if f['name']==fold_name)
    result=[]
    for key,role in [('source_train_ids','train'),('source_validation_ids','validation'),
                     ('target_development_ids','target_development')]:
        result.append([{**by_id[i],'original_partition':by_id[i]['partition'],'partition':role} for i in fold[key]])
    assert_source_partitions(result[0],result[1])
    return result
