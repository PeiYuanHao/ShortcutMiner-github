"""One bounded Full reanalysis of the previously inspected Pet pool."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
from experiments.eswa_20261019 import run_one


def main():
    original=run_one.arm.restrict_request
    def restrict(request,variant):
        request=original(request,variant);value=request.model_dump(mode='json')
        value['metadata']['real_case_scope']={'previously_inspected':True,'independent_confirmation':False,
            'historical_checkpoint_reuse':'Exact training bytes/labels/order, implementation, initialization and per-profile recipe verified.',
            'semantic_preservation':'Not human certified. Missing local segmentation/text tools require deferral.'}
        return type(request).model_validate(value)
    run_one.arm.restrict_request=restrict
    study=ROOT/'results_eswa_single/extensions/pet-v1'
    sys.argv=[str(ROOT/'experiments/eswa_20261019/run_one.py'),'--study-root',str(study),'--case-id','r-01',
        '--variant','full','--phase','formal','--profile',str(study/'protocol/profile.json'),
        '--shared-training-cache-root',str(ROOT/'results_eswa_single/shared_classifiers'),
        '--token-budget','600000','--attempt','1','--compact-role-prompts','--reserve-missing-text-usage',
        '--qualitative-sample-cap','12','--qualitative-task-cap','1']
    run_one.main()


if __name__=='__main__':main()
