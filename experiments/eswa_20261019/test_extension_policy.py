import json
from pathlib import Path
import pytest
from experiments.eswa_20261019.extension_policy import mask_feedback,mask_messages,WITHHELD
from experiments.eswa_20261019.extension_queue import classify
from experiments.eswa_20261019 import monitor


def test_feedback_mask_preserves_safety_ids_numbers_and_proposals():
    review={'status':'failed','task_preservation':'damaged','mechanism_specificity':'unsupported',
        'recommended_action':'revise','reviewed_pair_ids':['P1'],'task_content_observation':'Lost subject',
        'limitations':['No human verification'],'numeric_damage':0.2}
    payload={'candidate':{'hypothesis':'A frozen original hypothesis'},'proposal':{'rationale':'Original proposal'},
        'feedback':[{'quality_review_requires_revision':review,'prior_proposal':{'rationale':'Keep old operation'}}],
        'visual_review':review,'response_schema':{'properties':{'summary':{'type':'string'}}}}
    masked,paths=mask_feedback(payload)
    assert paths and masked['visual_review']['task_content_observation']==WITHHELD
    for key in ('status','task_preservation','mechanism_specificity','recommended_action','reviewed_pair_ids','numeric_damage'):
        assert masked['visual_review'][key]==review[key]
    assert masked['candidate']==payload['candidate'] and masked['proposal']==payload['proposal']
    assert masked['feedback'][0]['prior_proposal']['rationale']=='Keep old operation'
    assert masked['response_schema']==payload['response_schema']
    assert payload['visual_review']['task_content_observation']=='Lost subject'


def test_transmitted_text_matches_masked_payload_and_keeps_system_and_images():
    payload={'visual_review':{'status':'supported','rationale':'Detailed semantic feedback'}}
    messages=[{'role':'system','content':'Protocol'}, {'role':'user','content':json.dumps(payload)},
        {'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,abc'}}]}]
    masked=mask_messages(messages)
    assert json.loads(masked[1]['content'])==mask_feedback(payload)[0]
    assert masked[0]==messages[0] and masked[2]==messages[2]


def test_backend_failure_is_not_hidden_by_prior_model_correction(tmp_path):
    calls=tmp_path/'run/model_calls';calls.mkdir(parents=True)
    (calls/'failure.json').write_text(json.dumps({'failure_category':'protocol_schema_failed','completed_at':'2026-10-20'}))
    assert classify(tmp_path,1,'mandatory_quantitative_backend_execution_failed')=='ENGINEERING_ERROR'
    assert classify(tmp_path,1,'local file missing')=='ENGINEERING_ERROR'
    assert classify(tmp_path,1,'Model JSON did not satisfy the required protocol schema.')=='MODEL_PROTOCOL_FAILURE'


def test_monitor_extension_roots_cannot_escape_study(tmp_path,monkeypatch):
    monkeypatch.setattr(monitor,'STUDY',tmp_path)
    row={'case_id':'opaque','variant':'full','phase':'formal','study_root':str(tmp_path/'extensions/test')}
    assert monitor.arm_directory(row)==tmp_path/'extensions/test/audits/formal/opaque/full/attempt-01'
    with pytest.raises(ValueError):monitor.arm_directory({**row,'study_root':str(tmp_path.parent)})
