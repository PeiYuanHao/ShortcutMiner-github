from datetime import datetime,timezone,timedelta
import json
from experiments.eswa_20261019.formal_queue import can_start,schedule,failure_kind,consumed,CAP,HEADROOM,ENVELOPE


def test_cost_retains_timeout_telemetry_and_avoids_cumulative_double_count(tmp_path):
    first=tmp_path/'audits/development/case/full/attempt-01';first.mkdir(parents=True)
    (first/'EXECUTION_RECEIPT.json').write_text(json.dumps({'known_tokens':100,'new_tokens':100}))
    second=first.parent/'attempt-02';second.mkdir()
    (second/'EXECUTION_RECEIPT.json').write_text(json.dumps({'known_tokens':150,'new_tokens':50,'reserved_unknown_tokens':10}))
    third=tmp_path/'audits/formal/case/full/attempt-01/run/model_calls';third.mkdir(parents=True)
    (third/'telemetry.json').write_text(json.dumps({'total_tokens':20}))
    assert consumed(tmp_path)==(170,10)


def test_start_reserves_whole_audit_and_export_window():
    now=datetime.now(timezone.utc)
    assert can_start(now,now+timedelta(days=2),1000000,2000)
    assert not can_start(now,now+timedelta(hours=13),0,0)
    assert not can_start(now,now+timedelta(days=2),ENVELOPE-CAP-HEADROOM+1,0)
    assert not can_start(now,now+timedelta(days=2),ENVELOPE-CAP-HEADROOM-100,101)


def test_schedule_retains_private_order_without_truth_in_worker_rows():
    public={'cases':[{'case_id':'opaque2'},{'case_id':'opaque1'}]}
    private={cid:{'spec':{'config_execution_order':['Single-Agent','Fixed-Pipeline','Full'],
        'mechanism':'hidden','label_cue_match_rho':0.95}} for cid in ('opaque1','opaque2')}
    rows=schedule(public,private)
    assert [r['variant'] for r in rows[:3]]==['single_agent','fixed_pipeline','full']
    assert [r['case_id'] for r in rows]==['opaque2']*3+['opaque1']*3
    assert 'hidden' not in json.dumps(rows) and '0.95' not in json.dumps(rows)


def test_failure_classification_does_not_call_backend_errors_scientific_nulls(tmp_path):
    assert failure_kind(tmp_path,1)=='ENGINEERING_ERROR'
    directory=tmp_path/'run/model_calls';directory.mkdir(parents=True)
    (directory/'failure.json').write_text(json.dumps({'failure_category':'budget_exceeded'}))
    assert failure_kind(tmp_path,1)=='RESOURCE_LIMIT'
    assert failure_kind(tmp_path,124)=='TIME_LIMIT'
