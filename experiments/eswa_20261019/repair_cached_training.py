"""One zero-API engineering replay of failed training artifact publication; never an audit retry."""
import json
from pathlib import Path
import sys
import time
import argparse

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
from experiments.eswa_20261019.run_one import cache_identity
from experiments.eswa_20261019.resource_policy import publish_probe_cache
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.waterbirds_paper import sha,source_identity,archive_source,initialization
from shortcutminer.task_models.torchvision_matrix import TorchvisionResNetMatrix
from shortcutminer.tooling import build_default_tool_registry,ToolCall
from shortcutminer.tooling.base import ToolExecutionContext
from shortcutminer.experiment.real_executor import RealExperimentExecutor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1)
    attempt=parser.parse_args().attempt
    study=ROOT/'results_eswa_single'
    prior=study/'audits/development/dev-03/single_agent/attempt-04'
    old=prior/'dev-03-single-agent'
    output=study/('engineering/cached-training-publication-repair' if attempt==1 else f'engineering/cached-training-publication-repair-{attempt:02d}')
    output.mkdir(parents=True,exist_ok=False)
    identity=source_identity(require_clean=False)
    write(output/'source-identity.json',{'source':identity,'archive_sha256':archive_source(output/'source.zip',identity['source_tree_sha256'])})
    events=[json.loads(line) for line in (old/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    event=next(e for e in events if e['event']=='TOOL_STARTED' and e['details'].get('experiment_type')=='task_model_training')
    detail=event['details']
    call=ToolCall(tool_call_id=detail['tool_call_id'],tool_name='run_experiment',agent_name='quantitative',
        run_id='dev-03-single-agent',candidate_id=detail['candidate_id'],task_id=detail['task_id'],
        plan_step_id=detail['plan_step_id'],planning_check=detail['planning_check'],
        experiment_type=detail['experiment_type'],arguments=detail['arguments'])
    original=TorchvisionResNetMatrix.fit_matrix
    def fit(model,paths,labels,val_paths,val_labels,directory,fingerprint,*,probe_callback=None,shared_cache=None):
        init={arch:sha(initialization(arch)[0]) for arch in model.options.architectures}
        key,content=cache_identity(paths,labels,val_paths,val_labels,model.options.model_dump(mode='json'),init)
        cache=study/'shared_classifiers'/key
        if json.loads((cache/'TRAINING_IDENTITY.json').read_text())!=content:raise ValueError('Training cache mismatch')
        for arch in model.options.architectures:
            for seed in model.options.seeds:
                if not (cache/f'{arch}-seed{seed}/profile.json').is_file():raise ValueError('No completed weights; retraining forbidden in repair')
        write(output/'CACHE_REUSE.json',{'identity':key,'retraining_allowed':False,'original_failed_attempt':str(prior)})
        result=original(model,paths,labels,val_paths,val_labels,directory,fingerprint,probe_callback=probe_callback,shared_cache=cache)
        publish_probe_cache(model,cache,shared_cache)
        return result
    TorchvisionResNetMatrix.fit_matrix=fit
    run=output/call.run_id;run.mkdir()
    context=ToolExecutionContext(run_id=call.run_id,agent_name='quantitative',runs_root=output,
        dataset_root=prior/'dev-03-single-agent-input',model_capabilities=frozenset({'retraining'}),
        allowed_experiment_types=frozenset({'task_model_training'}),max_selected_samples=100000)
    started=time.monotonic()
    result=build_default_tool_registry(experiment_executor=RealExperimentExecutor()).execute(call,context,timeout_seconds=None)
    write(output/'TOOL_RESULT.json',result.model_dump(mode='json'))
    write(output/'REPAIR_RECEIPT.json',{'status':result.status,'wall_seconds':time.monotonic()-started,
        'api_calls':0,'new_training_runs':0,'scope':'Engineering-only replay of the exact failed tool; original audit remains failed and over quota.',
        'original_known_tokens':835460,'original_cumulative_token_stop':833210})
    if result.status!='success':raise RuntimeError(result.error)


if __name__=='__main__':main()
