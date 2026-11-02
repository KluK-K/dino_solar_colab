"""Explicit research confirmation workflow; no default test opening or model choice."""
import argparse
from pathlib import Path
from elresearch.common import ROOT,save_json
from elresearch.confirmation_workflow import prepare,freeze_preparation,execute,audit,specification_template


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    template=sub.add_parser('template');template.add_argument('path')
    prep=sub.add_parser('prepare');prep.add_argument('specification');prep.add_argument('destination')
    freeze=sub.add_parser('freeze');freeze.add_argument('prepared_plan')
    proposal=sub.add_parser('propose-examples');proposal.add_argument('destination')
    sub.add_parser('evaluate');sub.add_parser('audit');sub.add_parser('analyze');args=parser.parse_args()
    if args.action=='template':
        path=Path(args.path)
        if path.exists():raise FileExistsError('Never overwrite an existing scientific selection specification')
        save_json(path,specification_template());print('Unreviewed source-selection template saved; no registry frozen or test scored')
    elif args.action=='prepare':
        result=prepare(args.specification,args.destination);print('Prepared source-reviewed plan',result['preparation_id'])
    elif args.action=='freeze':
        result=freeze_preparation(args.prepared_plan);print('Immutable registry frozen',result['registry_id'],'no model scored yet')
    elif args.action=='evaluate':execute()
    elif args.action=='audit':print(audit())
    elif args.action=='propose-examples':
        from elresearch.confirmation_workflow import FINAL,local_path,propose_fixed_examples,BASE
        from elresearch import confirmation_registry as gate
        from elresearch.common import read_json,now
        if (FINAL/'frozen_registry.json').exists() or (FINAL/'observation_opened.json').exists():
            raise RuntimeError('Gallery selection is already frozen; do not replace it')
        destination=local_path(args.destination)
        if destination.exists():raise FileExistsError('Preserve the existing metadata-only gallery proposal')
        cohorts=[gate.cohort_binding(name) for name in sorted(gate.ALLOWED_COHORTS)]
        selected={}
        for cohort in cohorts:
            rows=[r for r in read_json(gate.check_file(cohort['manifest']))['records'] if r['partition']==cohort['partition']]
            selected[cohort['name']]=propose_fixed_examples(rows)
        save_json(destination,{'at':now(),'cohort_bindings':cohorts,'fixed_localization_examples':selected,
            'selection':'One per provided acquisition-channel/class when six groups; otherwise up to3 evenly spaced IDs per class. No pixels, masks or model scores read.',
            'status':'Prospective metadata-only proposal, not reviewed model choices or an inference registry. No guarantee of representative examples.'})
        print('Metadata-only fixed gallery proposed; no target model inference or observation marker')
    else:
        from elresearch.confirmation_analysis import analyze
        result=analyze();print('All frozen primary contrasts reported',len(result['primary_comparisons']),result['analysis_id'])

if __name__=='__main__':main()
