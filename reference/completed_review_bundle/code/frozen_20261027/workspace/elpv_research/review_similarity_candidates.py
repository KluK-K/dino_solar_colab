"""Native-resolution review sheets and measurements; no model calls or split edits."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from elresearch.common import read_json,save_json,sha256,now
from elresearch.near_duplicate_audit import FOLDER,image_path,d4,correlation


def main():
    result=read_json(FOLDER/'result.json');records=read_json(FOLDER/'input_records.json')['records'];by={r['id']:r for r in records}
    pairs=result['strong_similarity_pairs'];measurements=[]
    for page,start in enumerate(range(0,len(pairs),5),1):
        selected=pairs[start:start+5];fig,axes=plt.subplots(len(selected),3,figsize=(11,3.6*len(selected)),squeeze=False)
        for row,pair in enumerate(selected):
            a,b=by[pair['first_id']],by[pair['second_id']]
            with Image.open(image_path(a)) as image:first=np.asarray(image.convert('L')).copy()
            with Image.open(image_path(b)) as image:second=list(d4(np.asarray(image.convert('L')).copy()))[pair['orientation']]
            common=(min(first.shape[1],second.shape[1]),min(first.shape[0],second.shape[0]))
            first=np.asarray(Image.fromarray(first).resize(common,Image.Resampling.LANCZOS),float)
            second=np.asarray(Image.fromarray(second).resize(common,Image.Resampling.LANCZOS),float)
            slope=np.mean((first-first.mean())*(second-second.mean()))/np.var(second);intercept=first.mean()-slope*second.mean()
            difference=first-(slope*second+intercept)
            def target_boxes(r):return sorted((x['class'],tuple(x['normalized_xyxy'])) for x in r.get('boxes',[]) if x['class'] in {'crack','star_crack'})
            metrics={'first_id':a['id'],'second_id':b['id'],'native_common_shape':list(first.shape),
                'correlation_native':correlation(first,second),
                'highpass_correlation_native_sigma2':correlation(first-gaussian_filter(first,2),second-gaussian_filter(second,2)),
                'positive_affine_slope':float(slope),'affine_intercept':float(intercept),
                'affine_residual_RMSE_gray_units':float(np.sqrt(np.mean(difference**2))),
                'same_target_boxes':target_boxes(a)==target_boxes(b),'first_target_box_count':len(target_boxes(a)),
                'second_target_box_count':len(target_boxes(b)),
                'annotation_targets':[a['target'],b['target']],'audit_partitions':[a['audit_partition'],b['audit_partition']]}
            measurements.append(metrics)
            for column,(values,title) in enumerate([(first,a['id']),(second,b['id']),(difference,'Affine-adjusted difference, fixed ±8 gray units')]):
                axes[row,column].imshow(values,cmap='gray' if column<2 else 'coolwarm',vmin=0 if column<2 else -8,vmax=255 if column<2 else 8)
                axes[row,column].set_title(title,fontsize=8);axes[row,column].axis('off')
            axes[row,0].text(0,-.04,f"Native NCC {metrics['correlation_native']:.6f}; residual RMSE {metrics['affine_residual_RMSE_gray_units']:.3f}",transform=axes[row,0].transAxes,fontsize=7)
        fig.tight_layout();fig.savefig(FOLDER/f'native_candidates_page{page}.png',dpi=165,facecolor='white');plt.close(fig)
    save_json(FOLDER/'native_resolution_measurements.json',{'at':now(),'screen_result_sha256':sha256(FOLDER/'result.json'),
        'review_script_sha256':sha256(__file__),'pairs':measurements,'scope':'Data-origin diagnostics, not model output or automatic origin adjudication.'})
    print('Native review sheets:',len(pairs),'pairs; box-coverage agreement',sum(r['same_target_boxes'] for r in measurements),flush=True)


if __name__=='__main__':main()
