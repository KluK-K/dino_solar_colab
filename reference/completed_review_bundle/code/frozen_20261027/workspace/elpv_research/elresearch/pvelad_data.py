"""Audit the exact public PVEL-AD archive; missing test annotations never become normal labels."""
from collections import Counter
from pathlib import Path,PurePosixPath
import xml.etree.ElementTree as ET

from .common import ROOT,digest,now,read_json,save_json,sha256


ROOT_IN_ARCHIVE='solar_cell_EL_image/PVELAD/EL2021'
CLASS_NAMES={'crack','finger','black_core','thick_line','star_crack','corner','fragment','scratch',
             'horizontal_dislocation','vertical_dislocation','printing_error','short_circuit'}


def read_annotation(path):
    path=Path(path);text=path.read_text()
    if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():raise ValueError('DTD/entity declaration is not permitted')
    root=ET.fromstring(text)
    width,height=int(root.findtext('size/width')),int(root.findtext('size/height'))
    filename=root.findtext('filename')
    if not filename or Path(filename).name!=filename:raise ValueError('Unexpected image basename')
    boxes=[]
    for obj in root.findall('object'):
        name=obj.findtext('name')
        if name not in CLASS_NAMES:raise ValueError('Unmapped PVEL class: '+str(name))
        coordinates=[float(obj.findtext('bndbox/'+key)) for key in ['xmin','ymin','xmax','ymax']]
        if any(not value.is_integer() for value in coordinates):raise ValueError('Noninteger VOC coordinate')
        xmin,ymin,xmax,ymax=map(int,coordinates)
        if not (1<=xmin<=xmax<=width and 1<=ymin<=ymax<=height):raise ValueError('Invalid 1-based inclusive VOC bounds')
        boxes.append({'class':name,'raw_voc_xyxy':[xmin,ymin,xmax,ymax],
            'normalized_xyxy':[(xmin-1)/width,(ymin-1)/height,xmax/width,ymax/height],
            'difficult':int(obj.findtext('difficult') or 0),'truncated':int(obj.findtext('truncated') or 0)})
    if not boxes:raise ValueError('Empty XML requires an explicit normal-label policy')
    names={b['class'] for b in boxes}
    return {'filename':filename,'width':width,'height':height,'boxes':boxes,
        'target':int(bool(names&{'crack','star_crack'})),
        'target_line_crack':1 if 'crack' in names else (None if 'star_crack' in names else 0),
        'label_origin':'author trainval VOC boxes; crack or star_crack positive',
        'coordinate_convention':'VOC 1-based inclusive integer boxes -> normalized outer xyxy edges'}


def metadata_audit():
    archive=ROOT/'data/pvelad/archive';catalog=read_json(archive/'catalog.json')['items']
    archive_manifest=read_json(archive/'manifest.json')
    by_path={item['Path']:item for item in catalog};invalid_paths=[]
    for member in by_path:
        path=PurePosixPath(member.replace('\\','/'))
        if path.is_absolute() or '..' in path.parts:invalid_paths.append(member)
    if invalid_paths:raise ValueError('Unsafe paths in archive')
    base=ROOT/'data/pvelad/raw/extracted'/ROOT_IN_ARCHIVE
    paths=sorted((base/'trainval/Annotations').glob('*.xml'))
    if len(paths)!=4500:raise ValueError('Expected metadata extraction of all 4500 XML members')
    rows=[];errors=[]
    for path in paths:
        try:
            parsed=read_annotation(path)
            image_member=f"{ROOT_IN_ARCHIVE}/trainval/JPEGImages/{parsed['filename']}"
            if image_member not in by_path:raise ValueError('XML image has no archive member')
            xml_member=f'{ROOT_IN_ARCHIVE}/trainval/Annotations/{path.name}'
            stem=Path(parsed['filename']).stem
            rows.append({'id':'pvelad/'+stem,'origin_id':stem,'name':parsed['filename'],
                'dataset':'pvelad','upstream_split':'trainval','role':'annotated_external_candidate',
                'image_member':image_member,'image_bytes':int(by_path[image_member]['Size']),
                'image_member_crc':by_path[image_member].get('CRC'),
                'annotation_path':str(path.relative_to(ROOT)),'annotation_member':xml_member,
                'annotation_sha256':sha256(path),'annotation_type':'VOC_bounding_boxes',
                'module_id':None,'acquisition_id':None,
                'group_semantics':'physical acquisition/module identity not provided; no inferred physical grouping',
                **parsed})
        except Exception as error:errors.append({'annotation':path.name,'reason':str(error)})
    good=[];test=[];unknown=[]
    for member,item in by_path.items():
        if not member.lower().endswith('.jpg'):continue
        filename=PurePosixPath(member).name;stem=PurePosixPath(member).stem
        row={'id':'pvelad/'+stem,'origin_id':stem,'name':filename,'image_member':member,
             'image_bytes':int(item['Size']),'image_member_crc':item.get('CRC'),
             'dataset':'pvelad','module_id':None,'acquisition_id':None}
        if '/othertypes/good/' in member:
            good.append({**row,'target':0,'target_line_crack':0,'role':'author_reported_normal_candidate',
                'upstream_split':'othertypes/good','annotation_type':'author_directory_label',
                'label_origin':'author archive directory good; no pixel or box annotations'})
        elif '/test/JPEGImages/' in member:test.append(row['id'])
        elif '/othertypes/' in member:unknown.append(row['id'])
    objects=Counter(b['class'] for row in rows for b in row['boxes'])
    image_classes=Counter(name for row in rows for name in {b['class'] for b in row['boxes']})
    signature=digest({'archive_sha256':archive_manifest['archive_sha256'],'rows':rows,
        'good':good,'errors':errors,'audit_code_sha256':sha256(__file__)})
    result={'dataset':'pvelad_public_20260130_metadata','signature':signature,'created_at':now(),
        'archive_sha256':archive_manifest['archive_sha256'],'archive_bytes':archive_manifest['expected_bytes'],
        'download_source':'https://github.com/binyisu/PVEL-AD','file_id':archive_manifest['file_id'],
        'annotated_images':len(rows),'annotated_crack_positive':sum(r['target'] for r in rows),
        'line_crack_positive_images':sum(r['target_line_crack']==1 for r in rows),
        'star_only_images_excluded_from_line_only_sensitivity':sum(r['target_line_crack'] is None for r in rows),
        'reported_normal_images':len(good),'unlabeled_official_test_images':len(test),
        'unmapped_other_images':len(unknown),'all_jpeg_images':sum(x['Path'].lower().endswith('.jpg') for x in catalog),
        'object_class_counts':dict(objects),'image_class_counts':dict(image_classes),
        'metadata_shape_counts':{f'{w}x{h}':sum(r['width']==w and r['height']==h for r in rows)
                                for w,h in sorted({(r['width'],r['height']) for r in rows})},
        'errors':errors,'records':rows,'normal_candidates':good,
        'evaluation_state':'sealed_pending_image_audit','publication_eligible':False,
        'limits':['This exact public archive contains 4500 trainval XML files and no official-test XML labels, despite a public announcement about test labels.',
                  'Use author trainval only as an external cohort unused by our models; do not call this the official PVEL test benchmark.',
                  'Anomaly-free status of good images is an author directory label, not independently verified exhaustive annotation.',
                  'Do not label 19150 official-test images or 1540 meterial images negative from absence of XML.',
                  'Physical module/acquisition identifiers are unavailable; filename-adjacency blocks can only support an explicitly labeled sensitivity analysis.',
                  'Primary crack mapping includes line and star cracks; predeclared line-only sensitivity excludes star-only images.',
                  'Raw images are not redistributed; author data-use instructions remain attached.'],
        'next':'Extract selected images; verify image/annotation identity, dimensions and source/target duplicate leakage before registering final confirmation.'}
    dest=ROOT/'data/pvelad/metadata_manifest.json'
    if dest.exists():
        previous=read_json(dest)
        if previous['signature']!=signature:raise RuntimeError('PVEL metadata manifest is immutable')
        return previous
    save_json(dest,result)
    save_json(ROOT/'artifacts/publication_20261017/pvelad_metadata_audit.json',
              {k:v for k,v in result.items() if k not in {'records','normal_candidates'}})
    return result


if __name__=='__main__':
    result=metadata_audit()
    print({k:result[k] for k in ['annotated_images','annotated_crack_positive','reported_normal_images',
        'unlabeled_official_test_images','unmapped_other_images','metadata_shape_counts','errors']},flush=True)
