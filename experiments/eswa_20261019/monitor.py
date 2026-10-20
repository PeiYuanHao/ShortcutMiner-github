"""Token-free local watcher; emit only a compact state to scheduled AI checks."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time

STUDY=Path(__file__).resolve().parents[2]/'results_eswa_single'


def read(path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return default


def write(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def alive(pid):
    if not pid:return False
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x100000,False,int(pid))
    if not handle:return False
    try:return kernel.WaitForSingleObject(handle,0)==258
    finally:kernel.CloseHandle(handle)


def attention_required(status,running,idle):
    return status in {'FAILED','PROCESS_EXITED_CHECK_RESULT'} or (status=='RUNNING' and running and idle>1200)


def queue_attention(stage,running):
    return 'requires_' in stage or (stage.endswith('_running') and not running)


def arm_directory(row):
    root=Path(row.get('study_root') or STUDY).resolve()
    root.relative_to(STUDY.resolve())
    return root/'audits'/row.get('phase','development')/row['case_id']/row['variant']/f"attempt-{int(row.get('attempt',1) or 1):02d}"


def inspect():
    live=read(STUDY/'LIVE_STATUS.json',{})
    tasks=[];current_tokens=0;current_calls=0
    for row in live.get('tasks',[]):
        attempt=int(row.get('attempt',1) or 1)
        arm=arm_directory(row)
        state=read(arm/'arm-status.json',{})
        receipt=read(arm/'EXECUTION_RECEIPT.json',{})
        telemetry=[read(p,{}) for p in arm.glob('*/model_calls/telemetry.json')]
        tokens=sum(t.get('total_tokens',0) or 0 for t in telemetry)
        calls=sum(t.get('total_calls',0) or 0 for t in telemetry)
        current_tokens+=receipt.get('new_tokens',receipt.get('known_tokens',tokens))
        current_calls+=calls
        pid=row.get('pid');running=alive(pid)
        status=state.get('status',row['status'])
        if row['status']=='FAILED_RETAINED':status='FAILED_RETAINED'
        # Completed rows cannot acquire a new live worker when Windows recycles a PID.
        if status!='RUNNING':running=False
        recent=[p.stat().st_mtime for p in arm.glob('*/events.jsonl')]
        recent.extend(p.stat().st_mtime for p in arm.glob('*/models/task/*/*/progress.json'))
        latest=max(recent,default=arm.stat().st_mtime if arm.exists() else time.time())
        idle=round(max(0,time.time()-latest))
        if status=='RUNNING' and not running:status='PROCESS_EXITED_CHECK_RESULT'
        tasks.append({'case':row['case_id'],'variant':row['variant'],'attempt':attempt,
            'status':status,'stage':state.get('stage'),'process_alive':running,
            'known_tokens':receipt.get('known_tokens',tokens),'calls':calls,
            'idle_seconds':idle,'needs_attention':attention_required(status,running,idle)})
    past=[]
    attempts=list((STUDY/'audits').glob('*/*/*/attempt-*'))
    attempts.extend((STUDY/'extensions').glob('*/audits/*/*/*/attempt-*'))
    for directory in attempts:
        path=directory/'EXECUTION_RECEIPT.json'
        if path.exists():
            r=read(path,{})
            past.append(r.get('new_tokens',r.get('known_tokens',0)))
        else:
            past.append(sum((read(p,{}).get('total_tokens',0) or 0) for p in directory.glob('*/model_calls/telemetry.json')))
    # Include live calls not yet represented by a final receipt.
    live_extra=0
    for row in live.get('tasks',[]):
        arm=arm_directory(row)
        if arm not in attempts and not (arm/'EXECUTION_RECEIPT.json').exists():
            live_extra+=sum((read(p,{}).get('total_tokens',0) or 0) for p in arm.glob('*/model_calls/telemetry.json'))
    start=read(STUDY/'STUDY_START.json',{})
    deadline=start.get('experiment_deadline_utc')
    expired=bool(deadline and datetime.now(timezone.utc)>=datetime.fromisoformat(deadline))
    queue_stage=live.get('stage','unknown');queue_alive=alive(live.get('queue_pid'))
    core={'queue_stage':queue_stage,'queue_alive':queue_alive,
        'formal_started':live.get('formal_started',False),'tasks':tasks,
        'cumulative_known_tokens':sum(past)+live_extra,'deadline_expired':expired,
        'needs_attention':any(t['needs_attention'] for t in tasks) or expired or queue_attention(queue_stage,queue_alive),
        'watcher_pid':os.getpid()}
    # Exclude idle time and token increments from change alerts; retain them in the snapshot.
    signature={'queue_stage':core['queue_stage'],'formal_started':core['formal_started'],
        'queue_alive':queue_alive,'needs_attention':core['needs_attention'],
        'states':[(t['case'],t['variant'],t['attempt'],t['status'],t['stage'],t['needs_attention']) for t in tasks],
        'deadline_expired':expired}
    core['signature']=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    core['updated_utc']=datetime.now(timezone.utc).isoformat()
    previous=read(STUDY/'MONITOR_SNAPSHOT.json',{})
    core['meaningful_change']=previous.get('signature')!=core['signature']
    reviewed=read(STUDY/'MONITOR_LAST_REVIEW.json',{})
    core['changed_since_last_review']=reviewed.get('signature')!=core['signature']
    write(STUDY/'MONITOR_SNAPSHOT.json',core)
    if core['meaningful_change']:
        with (STUDY/'MONITOR_EVENTS.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(core,ensure_ascii=False,separators=(',',':'))+'\n')
    return core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--acknowledge',action='store_true',help='Acknowledge this signature after actual review/action.')
    parser.add_argument('--interval',type=int,default=60)
    args=parser.parse_args()
    if not args.once:
        import msvcrt
        handle=(STUDY/'monitor.lock').open('a+b');handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
        try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('A local monitor is already active')
        write(STUDY/'MONITOR_PROCESS.json',{'pid':os.getpid(),'interval_seconds':args.interval,
            'started_utc':datetime.now(timezone.utc).isoformat(),'calls_language_models':False})
    while True:
        core=inspect()
        if args.once:
            if args.acknowledge:
                write(STUDY/'MONITOR_LAST_REVIEW.json',{'signature':core['signature'],'reviewed_utc':core['updated_utc']})
            print(json.dumps(core,ensure_ascii=False,separators=(',',':') if args.compact else None),flush=True)
            return
        if core['deadline_expired'] or (STUDY/'DELIVERY_COMPLETE.json').exists():return
        time.sleep(max(15,args.interval))


if __name__=='__main__':main()
