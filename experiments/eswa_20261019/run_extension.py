"""Controlled fixed-candidate extension of the shared ESWA harness."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
from experiments.eswa_20261019 import run_one
from experiments.eswa_20261019.extension_policy import mask_feedback,mask_messages,VERSION
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.waterbirds_paper import sha
from shortcutminer.experiment.candidate_validation import CandidateMeasurementPlan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--case-id',required=True);args=parser.parse_args()
    study=args.study_root.resolve()
    manifest=json.loads((study/'protocol/CONTROLLED_INPUTS.json').read_text())
    item=next(row for row in manifest['arms'] if row['case_id']==args.case_id)
    condition=item['condition'];source=Path(item['discovery_source'])
    for path,digest in item['frozen_phase_hashes'].items():
        if sha(Path(path))!=digest:raise ValueError('Fixed discovery/measurement input changed')
    source_identity=json.loads((source/'source_identity.json').read_text())
    if sha(source/'source_snapshot.zip')!=source_identity['archive_sha256']:raise ValueError('Discovery source archive changed')
    if json.loads((source/'run.json').read_text())['status']!='COMPLETED':raise ValueError('Discovery is incomplete')
    output=study/'audits/formal'/args.case_id/'full/attempt-01'
    arm=run_one.arm;original_execute=arm.legacy.execute
    def execute(request,*positional,**kwargs):
        if request.run_id.endswith('-discovery'):
            write(output/'FIXED_DISCOVERY_REUSE.json',{'condition':condition,'source':str(source),
                'fixed_phase_hashes':item['frozen_phase_hashes'],
                'scope':'Reused training-only discovery; no new independent discovery or dataset. Original source remains retained.'})
            return source
        return original_execute(request,*positional,**kwargs)
    arm.legacy.execute=execute
    measurement=Path(item['measurement_source'])
    def frozen_measurement(client,*,root,candidates,observations,receipt_path,prompt):
        saved=json.loads(measurement.read_text());plan=CandidateMeasurementPlan.model_validate(saved['plan'])
        ids=[candidate['candidate_id'] for candidate in candidates]
        if len(plan.measurements)!=len(ids) or {entry.candidate_id for entry in plan.measurements}!=set(ids):raise ValueError('Frozen candidate/measurement coverage mismatch')
        Path(receipt_path).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(measurement,receipt_path)
        return plan,saved
    arm.entry.freeze_candidate_measurements=frozen_measurement
    original_restrict=arm.restrict_request
    def restrict(request,variant):
        request=original_restrict(request,variant);value=request.model_dump(mode='json')
        value['metadata']['controlled_extension']={'condition':condition,'candidate_source_sha256':item['frozen_phase_hashes'][str(source/'identify/output.json')],
            'independent_discovery':False,'independent_confirmation':False,'feedback_mask_version':VERSION,
            'limitation':'Conditional comparison on reused training discovery and previously used held-out sources; not a new blind confirmation.'}
        return type(request).model_validate(value)
    arm.restrict_request=restrict
    BaseText=arm.TextClient;BaseVision=arm.VisionClient
    class Text(BaseText):
        def __init__(self,*a,**kw):
            if condition=='fixed_routing':kw['variant']='without_adaptive_routing'
            super().__init__(*a,**kw)
        def _invoke_structured(self,**kw):
            if condition=='without_natural_language_semantic_feedback' and kw['agent_name']!='identify':
                original=kw['payload_for_hash'];delivered,changes=mask_feedback(original)
                if changes:
                    write(output/'feedback_masks'/(uuid4().hex+'.json'),{'version':VERSION,'agent':kw['agent_name'],
                        'run_id':kw['run_id'],'masked_paths':changes,'original_payload':original,'delivered_payload':delivered,
                        'validator_bypassed':False,'categorical_safety_findings_retained':True})
                    kw['payload_for_hash']=delivered;kw['messages']=mask_messages(kw['messages'])
            return super()._invoke_structured(**kw)
    class Vision(BaseVision):
        def __init__(self,*a,**kw):
            if condition=='fixed_routing':kw['variant']='without_adaptive_routing'
            super().__init__(*a,**kw)
    arm.TextClient=Text;arm.VisionClient=Vision
    sys.argv=[str(ROOT/'experiments/eswa_20261019/run_one.py'),'--study-root',str(study),
        '--case-id',args.case_id,'--variant','full','--phase','formal','--profile',str(study/'protocol/profile.json'),
        '--shared-training-cache-root',str(ROOT/'results_eswa_single/shared_classifiers'),
        '--token-budget','600000','--attempt','1','--compact-role-prompts','--reserve-missing-text-usage',
        '--qualitative-sample-cap','12','--qualitative-task-cap','1']
    run_one.main()


if __name__=='__main__':main()
