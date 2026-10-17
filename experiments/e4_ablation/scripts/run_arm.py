"""Run one isolated E4 arm using the shared scientific harness (Windows)."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/experiments')]
import run_candidate_complete_image_audit as entry
import run_autonomous_image_audit as legacy
from experiments.e4_ablation.variants.policies import (
    VARIANTS, TextClient, VisionClient, ControllerSession, restrict_request, write,
)
from shortcutminer.protocol.model import ModelRuntimeConfigSet
from shortcutminer.tooling.runtime import RunToolBudgetState


class BranchOnlyCompleted(Exception): pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--training-input',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--runs-root',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--protocol-revision',choices=['legacy','metadata-contract-v2','descriptor-codec-v3'],default='legacy')
    args=parser.parse_args(); args.runs_root.mkdir(parents=True,exist_ok=True)
    state_path=args.runs_root/'arm-status.json'
    if state_path.exists() and json.loads(state_path.read_text(encoding='utf-8'))['status']=='COMPLETED':
        print('Arm already complete; preserved.',flush=True);return
    status={'variant':args.variant,'run_id':args.run_id,'status':'RUNNING','started_unix':time.time(),
        'framework':'shared typed harness; single shared controller only for single_agent',
        'protocol_revision':args.protocol_revision,
        'text_call_cap':200,'visual_call_cap':50,'tool_call_cap':150,
        'human_adjudication_available':False,'candidate_reports_from_other_arms':False}
    write(state_path,status)
    session=ControllerSession(args.runs_root/'single-controller-memory.json') if args.variant=='single_agent' else None
    clients={}
    original_loader=legacy.load_model_runtime_config
    def load_model(path):
        config=original_loader(path)
        cap=50 if config.capabilities.supports_image_input else 200
        return config.model_copy(update={'model_call_budget':config.model_call_budget.model_copy(update={
            'max_total_calls':cap,'max_calls_per_agent':cap})})
    legacy.load_model_runtime_config=load_model
    def text_factory(config,**kwargs):
        clients['text']=TextClient(config,variant=args.variant,session=session,protocol_revision=args.protocol_revision,**kwargs)
        return clients['text']
    def vision_factory(config,**kwargs):
        clients['vision']=VisionClient(config,variant=args.variant,session=session,protocol_revision=args.protocol_revision,**kwargs)
        return clients['vision']
    entry.EvidenceContextClient=text_factory
    legacy.OpenAICompatibleClient=vision_factory

    # Restore actual calls on process resume, including retries, across both audit phases.
    # Clients' telemetry persists all provider attempts; no budget reset on resume.
    def restore_provider_budgets():
        for name,client in clients.items():
            if getattr(client,'_e4_restored',False): continue
            for p in args.runs_root.glob('*/model_calls/*.json'):
                record=json.loads(p.read_text(encoding='utf-8'))
                if not record.get('invocation_id'):continue
                if record.get('model')!=client.config.model or record.get('status')=='budget_exceeded':continue
                client._budget_state.calls_total+=1
                agent=record.get('agent_name','unknown')
                client._budget_state.calls_by_agent[agent]=client._budget_state.calls_by_agent.get(agent,0)+1
                usage=record.get('usage',{})
                if usage.get('total_tokens') is not None:client._budget_state.known_total_tokens+=usage['total_tokens']
                elif not usage.get('available'):client._budget_state.unknown_usage_seen=True
            client._e4_restored=True

    original_build=legacy.build_local_tool_runtimes
    used=len(list(args.runs_root.glob('*/tools/*/*.json')))
    tool_budget=RunToolBudgetState(150,used=used)
    def build_tools(**kwargs):
        tools=original_build(**kwargs)
        for runtime in tools.values():runtime.run_budget_state=tool_budget
        return tools
    legacy.build_local_tool_runtimes=build_tools
    original_execute=legacy.execute
    def execute(request,*positional,**kwargs):
        restore_provider_budgets()
        request=restrict_request(request,args.variant)
        # Runtime identities match actual configured common ceilings, not role-specific hidden caps.
        request=request.model_copy(update={'model_runtime':ModelRuntimeConfigSet(
            shared=clients['text'].config,per_agent={'qualitative':clients['vision'].config})})
        status['stage']='discovery' if request.run_id.endswith('-discovery') else 'validation'
        status['current_directory']=str(args.runs_root/request.run_id)
        write(state_path,status)
        directory=original_execute(request,*positional,**kwargs)
        if args.variant=='without_quantitative':
            status['final_directory']=str(directory)
            raise BranchOnlyCompleted()
        status['final_directory']=str(directory)
        return directory
    legacy.execute=execute
    original_select=legacy.select_autonomous_construction
    def select(*positional,**kwargs):
        if args.variant=='without_qualitative':kwargs['disable_qualitative']=True
        if args.variant=='without_counterevidence':kwargs['seek_counterexamples']=False
        return original_select(*positional,**kwargs)
    legacy.select_autonomous_construction=select
    old_argv=sys.argv
    sys.argv=[str(Path(entry.__file__)), '--input',str(args.input),'--training-input',str(args.training_input),
        '--profile',str(args.profile),'--runs-root',str(args.runs_root),'--run-id',args.run_id,
        '--task',args.task,'--isolated-discovery','--representative-review','--validation']
    try:
        try:entry.main()
        except BranchOnlyCompleted:pass
        status['status']='COMPLETED'
    except BaseException as exc:
        status['status']='FAILED';status['error_type']=type(exc).__name__
        # No uncontrolled exception text or HTTP content is copied into study summaries.
        raise
    finally:
        sys.argv=old_argv
        status['finished_unix']=time.time();status['wall_seconds']=status['finished_unix']-status['started_unix']
        status['actual_provider_calls']={k:v._budget_state.calls_total for k,v in clients.items()}
        status['actual_tool_calls']=tool_budget.used
        write(state_path,status)


if __name__=='__main__':main()
