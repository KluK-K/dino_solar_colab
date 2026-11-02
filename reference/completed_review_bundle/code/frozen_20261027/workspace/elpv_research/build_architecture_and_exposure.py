"""Trace the implemented network and source/confirmation exposure as vector figures.

The CPU structural probe uses synthetic tokens/images and no pretrained model.
It verifies decoder dimensions/gradients, not scientific predictive accuracy.
"""
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch
from torch import nn

from elresearch.common import ROOT,read_json,save_json,sha256,now
from elresearch.encoders import geometry,patch_geometry
from elresearch.full_models import dense_loss
from elresearch.publication_dense_models import make_dense_dino
from elresearch.publication_normal_bank import OrientationNormalBank
from elresearch.confirmation_population import checked_policy

BASE=ROOT/'artifacts/publication_20261017'
FOLDER=BASE/'architecture_and_exposure_v1'
INK='#193e48';TEAL='#28788b';PURPLE='#7955a7';ORANGE='#b57b32';GRAY='#617b84'


class TokenFixture(nn.Module):
    """Shape-only stand-in with the official stored token dimensions/depths."""
    def __init__(self,config):
        super().__init__();self.config=SimpleNamespace(**config)
        self.patch=nn.Conv2d(3,config['hidden_size'],16,16)
        self.prefix=nn.Parameter(torch.randn(1,1+config['num_register_tokens'],config['hidden_size']))
        self.layer=nn.ModuleList([nn.Identity() for _ in range(config['num_hidden_layers'])])
        self.norm=nn.LayerNorm(config['hidden_size'])

    def forward(self,pixel_values):
        patches=self.patch(pixel_values).flatten(2).transpose(1,2)
        tokens=torch.cat([self.prefix.expand(len(pixel_values),-1,-1),patches],1)
        for layer in self.layer:tokens=layer(tokens)
        return SimpleNamespace(last_hidden_state=self.norm(tokens))


def structural_probe(config,source_run):
    torch.manual_seed(20261017);torch.set_num_threads(2)
    g=geometry(64,64,64);_,coords=patch_geometry(g,16)
    dim=config['hidden_size'];items=[]
    for index,group in enumerate(['fixture_A','fixture_B','fixture_query']):
        view={'cls':torch.randn(dim),'patch':torch.randn(16,dim),
              'coords':torch.from_numpy(coords.reshape(-1,2))}
        items.append({'id':f'fixture{index}','group':group,'views':[view]*8})
    bank=OrientationNormalBank(items,{'kind':'synthetic token-shape probe only'})
    images=torch.randn(1,3,64,64);valid=torch.ones(1,64,64,dtype=torch.bool)
    target=torch.zeros(1,64,64,dtype=torch.long);target[:,:,30:33]=1
    records=[]
    for reference in [False,True]:
        model=make_dense_dino(TokenFixture(config),'frozen',11,bank if reference else None)
        shapes={};handles=[]
        def hook(name):
            def capture(module,args,output):shapes[name]={'input':list(args[0].shape),'output':list(output.shape)}
            return capture
        for name,module in [('normal_projection',model.normal_projection),('fusion',model.fuse),
                            ('image_head',model.image_head)]+[(f'decoder_{i+1}',m) for i,m in enumerate(model.decoder)]:
            handles.append(module.register_forward_hook(hook(name)))
        output=model(images,valid,[g],[('fixture_query',0)])
        loss=dense_loss(output,target,torch.ones(1),torch.ones(3));loss.backward()
        for handle in handles:handle.remove()
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable==source_run['trainable_parameters']
        assert model.depths==[2,5,8,11] and model.fuse[0].in_channels==576
        assert model.normal_projection[0].in_channels==771
        assert list(output['segmentation'].shape)==[1,3,64,64] and list(output['image'].shape)==[1]
        assert shapes['image_head']['input']==[1,769]
        required=['normal_projection.0.weight','fuse.0.weight','segmentation.weight','image_head.linear.weight']
        parameters=dict(model.named_parameters())
        assert all(parameters[n].grad is not None and torch.isfinite(parameters[n].grad).all() for n in required)
        assert all(p.grad is None for p in model.encoder.parameters())
        records.append({'reference':reference,'fixture_input_size':64,'output_shapes':{k:list(v.shape) for k,v in output.items()},
            'intermediate_shapes':shapes,'decoder_trainable_parameters':trainable,
            'source_decoder_parameter_count_matches':True,'required_head_gradients_finite':True,
            'frozen_encoder_gradients_absent':True,
            'reference_eligible_groups':[x['group'] for x in bank.items if x['group']!='fixture_query'] if reference else []})
    return {'at':now(),'official_stored_config':config,'cases':records,
        'scope':'Synthetic shape/gradient probe using64px inputs and an identity-layer token fixture; no pretrained weights, real images, training fit, model-performance evidence or timing claim.'}


def box(ax,x,y,w,h,title,detail='',color=TEAL,fill='#eef6f7',size=10):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.016',
                         linewidth=1.1,edgecolor=color,facecolor=fill,zorder=3)
    ax.add_patch(patch)
    ax.text(x+w/2,y+h*(.64 if detail else .5),title,ha='center',va='center',
            fontsize=size,weight='bold',color=INK,zorder=4)
    if detail:ax.text(x+w/2,y+h*.28,detail,ha='center',va='center',fontsize=size-1.2,color=GRAY,linespacing=1.5,zorder=4)
    return (x,y,w,h)


def arrow(ax,start,end,color=GRAY,style='-',connection='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=10,
        linewidth=1.2,color=color,linestyle=style,connectionstyle=connection,
        shrinkA=2,shrinkB=3,zorder=2))


def route(ax,points,color=GRAY,style='-'):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.1,ls=style,zorder=2)
    arrow(ax,points[-2],points[-1],color,style)


def export(fig,name):
    for extension in ['png','pdf','svg']:fig.savefig(FOLDER/f'{name}.{extension}',dpi=235,facecolor='white')
    plt.close(fig)


def architecture(config,head_count):
    fig,ax=plt.subplots(figsize=(13.6,8.6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.015,.98,'Implemented DINOv3 joint inspection model',fontsize=17,weight='bold',color=INK,va='top')
    ax.text(.015,.93,'Whole-field classification and three-class segmentation share the encoder; supervision is explicitly matched.',fontsize=10.5,color=GRAY)
    box(ax,.015,.72,.145,.13,'EL field of view','336 × 336 letterbox\nGray → RGB; valid-area mask',size=9)
    box(ax,.205,.70,.18,.17,'DINOv3-S / 16','12 transformer blocks\n384-d tokens; 4 registers',color=PURPLE,fill='#f3f0f8',size=10)
    arrow(ax,(.16,.785),(.205,.785))
    box(ax,.435,.77,.22,.12,'Depths 3, 6, 9, 12','Remove CLS/registers\n4 × (21 × 21 × 384)',size=9.5)
    box(ax,.435,.59,.22,.12,'Final CLS + valid patch mean','384 + 384 dimensions\nOnly fully image-valid patches',size=9.5)
    arrow(ax,(.385,.81),(.435,.83),color=PURPLE)
    arrow(ax,(.385,.735),(.435,.65),color=PURPLE)
    box(ax,.71,.77,.26,.12,'Multi-depth projections','4 × Conv1×1: 384 → 128\n512 projected channels',size=10)
    arrow(ax,(.655,.83),(.71,.83))
    box(ax,.71,.57,.26,.14,'Image score','LayerNorm on 768 features\nConcat local evidence → Linear(769,1)',color=PURPLE,fill='#f3f0f8',size=9.5)
    arrow(ax,(.655,.65),(.71,.65),color=PURPLE)
    box(ax,.015,.34,.20,.18,'Normal references (optional)','≤256 selected training images\nAnnotated-normal patches only\nSame D4 view; query family excluded',color=ORANGE,fill='#faf5e9',size=9)
    box(ax,.265,.34,.24,.18,'Similarity-matched residual slot','Retrieve up to 5 similar images\nCosine distance + 0.1 × squared position\n[q, q − r, x, y, cosine distance]',color=ORANGE,fill='#faf5e9',size=9)
    arrow(ax,(.215,.43),(.265,.43),color=ORANGE)
    route(ax,[(.295,.70),(.295,.555),(.385,.555),(.385,.52)],color=PURPLE)
    box(ax,.555,.35,.15,.16,'Local projection','771 → 64 channels\nNo-reference arm:\nresidual = distance = 0',color=ORANGE,fill='#faf5e9',size=9)
    arrow(ax,(.505,.43),(.555,.43),color=ORANGE)
    box(ax,.76,.34,.21,.17,'Shared decoder input','512 + 64 → ConvBlock(128)\nIdentical residual-slot network\nin reference and no-reference arms',size=9)
    arrow(ax,(.705,.43),(.76,.43),color=ORANGE)
    route(ax,[(.97,.83),(.997,.83),(.997,.425),(.97,.425)],color=TEAL)
    box(ax,.21,.10,.25,.17,'Four upsampling/detail stages','21 → 42 → 84 → 168 → 336\nWidths 96 → 64 → 32 → 24\nNormalized grayscale detail at each stage',size=9)
    route(ax,[(.865,.34),(.865,.315),(.335,.315),(.335,.27)])
    route(ax,[(.015,.785),(.001,.785),(.001,.185),(.21,.185)],style='--')
    box(ax,.515,.10,.19,.17,'Three-class pixel logits','Normal / crack / other defect\nKnown pixels supervise CE + Dice\nUndo letterbox for native evaluation',size=9)
    arrow(ax,(.46,.185),(.515,.185))
    box(ax,.76,.10,.21,.17,'Local crack evidence','Mean top 5% image-area\ncrack-vs-other log-odds\nFeeds the image-score head',color=PURPLE,fill='#f3f0f8',size=9)
    arrow(ax,(.704,.185),(.76,.185),color=PURPLE)
    route(ax,[(.97,.185),(.985,.185),(.985,.63),(.97,.63)],color=PURPLE)
    ax.text(.015,.055,f'Frozen decoder/head: {head_count:,} trainable parameters. LoRA adds 49,152 Q/V parameters in the last four blocks.',fontsize=9,color=GRAY)
    ax.text(.015,.017,'Live LoRA/full encoders do not use stale reference banks. Post-LoRA references require merge → freeze → re-encode → rebuild memory.',fontsize=9,color=GRAY)
    fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.02)
    export(fig,'implemented_dino_joint_architecture')


def exposure(primary):
    fig,ax=plt.subplots(figsize=(13.6,8.2));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.015,.98,'Data and model-selection exposure',fontsize=17,weight='bold',color=INK,va='top')
    ax.text(.015,.925,'A new sample can be group-disjoint while its dataset has already influenced hyperparameter selection.',fontsize=10.5,color=GRAY)
    box(ax,.015,.68,.24,.19,'UCF source development','430 train + 144 validation\nRates / regularization / checkpoints\nImage labels or masks by experiment',color=ORANGE,fill='#faf5e9',size=10)
    box(ax,.015,.34,.24,.22,'SCDD observed development','351 train + 120 validation\n5 / 10 / 20 / 100% train-label budgets\nThree acquisition-held-out directions\nThree paired fitted seeds',size=10)
    arrow(ax,(.135,.68),(.135,.56),color=ORANGE,style='--')
    ax.text(.15,.61,'Upstream selection',fontsize=8.5,color=ORANGE)
    box(ax,.34,.40,.25,.35,'One final frozen registry','Weights; source thresholds / temperatures\nCode/runtime/data hashes; label exposure\nPrimary contrasts and endpoints\nFixed examples; near-copy population policy\nSource / cost / ELPV gates finish first',color=PURPLE,fill='#f3f0f8',size=10)
    arrow(ax,(.255,.76),(.34,.65),color=ORANGE)
    arrow(ax,(.255,.46),(.34,.51),color=TEAL)
    box(ax,.69,.73,.285,.15,'SCDD confirmation: 117 images','Provided capture-family-disjoint test\nSame acquisition channels; physical IDs unverified',size=10)
    box(ax,.69,.49,.285,.17,'UCF confirmation: 507 images','Sample/group-disjoint from source574\nUCF already exposed through source selection\nAuthor-policy real provenance; file-level uncertain',color=ORANGE,fill='#faf5e9',size=9.5)
    box(ax,.69,.21,.285,.20,f"PVEL primary view: {primary['images']:,} images",f"{primary['positive']:,} known crack + {primary['negative']:,} author normal\nScore original 12,326-image pool once\nPrimary / full-pool / broader-screen summaries\nNo verified acquisition-group intervals",size=9.5)
    for y in [.805,.575,.31]:arrow(ax,(.59,.575),(.69,y),color=PURPLE)
    box(ax,.015,.10,.575,.12,'Separate ELPV supplement','1,968 course-development images; 4 outer folds; nested HOG / DINOv3 Logistic / Linear / RBF\nExpert review label differs from crack truth; 656 course-test images remain unopened',color=GRAY,fill='#f3f6f7',size=9)
    ax.text(.015,.067,'Both UCF and SCDD development outcomes inform the study design; per-model source-only fitting does not make either dataset wholly unseen to the research process.',fontsize=8.1,color=GRAY)
    ax.text(.015,.038,'Training references never include target images. Target pixels were reviewed for integrity; target model fitting or threshold calibration remains prohibited.',fontsize=8.1,color=GRAY)
    ax.text(.015,.009,'Training-label fractions exclude additional validation/HPO labels; these simulated budgets do not measure prospective annotation-time savings.',fontsize=8.1,color=GRAY)
    fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.02)
    export(fig,'source_selection_and_confirmation_exposure')


def main():
    FOLDER.mkdir(exist_ok=True)
    source_path=BASE/'dense_development_v1/calibration/dinov3_dense_frozen_lr0.0003_seed11/run.json'
    source=read_json(source_path)
    actual=source['encoder']['config']
    config={key:actual[key] for key in ['hidden_size','patch_size','num_register_tokens','num_hidden_layers']}
    assert config=={'hidden_size':384,'patch_size':16,'num_register_tokens':4,'num_hidden_layers':12}
    policy_path=BASE/'confirmation_preparation_v1/cohort_analysis_policy_v1.json'
    policy=checked_policy({'path':str(policy_path.relative_to(ROOT)),'sha256':sha256(policy_path)})
    primary=policy['cohorts']['pvelad_restricted_confirmation_v2']['subsets']['primary']
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    probe=structural_probe(config,source)
    architecture(config,source['trainable_parameters']);exposure(primary)
    files=['full_models.py','publication_dense_models.py','publication_normal_bank.py','publication_dense_data.py',
           'publication_models.py','adapt.py','encoders.py','confirmation_population.py','confirmation_workflow.py']
    bindings={'at':now(),'generator_sha256':sha256(__file__),'scientific_sources':{f:sha256(ROOT/'elresearch'/f) for f in files},
        'source_model_run':{'path':str(source_path.relative_to(ROOT)),'sha256':sha256(source_path)},
        'population_policy_sha256':sha256(policy_path),'structural_probe':probe,
        'recorded_dimensions':{'input_size':336,'patch_grid':21,'intermediate_depths_1_based':[3,6,9,12],
            'projected_depth_channels':[128]*4,'normal_input_channels':771,'normal_output_channels':64,
            'fusion_input_channels':576,'upsample_channels':[96,64,32,24],'segmentation_classes':3,
            'image_head_dimensions':769,'LoRA_trainable_encoder_parameters':49152},
        'limits':'Method/protocol figures and synthetic shape verification; no model-performance or novelty evidence. No pretrained weights loaded or real images opened by this script.'}
    save_json(FOLDER/'source_and_probe.json',bindings)
    print('Exported implemented architecture/exposure figures and synthetic dimension/gradient checks',flush=True)


if __name__=='__main__':main()
