import csv
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image

from shortcutminer.experiment.intervention_alignment import binding_contract, validate_operation, measure_preview_target
from shortcutminer.experiment.autonomous_construction import select_autonomous_construction, preview_construction
from shortcutminer.intervention.protocol import GlobalColorTransform
from shortcutminer.experiment.channel_ratio_calibration import ChannelRatioCalibrationChoice,calibrate_channel_ratio
from shortcutminer.intervention.channel_ratio import transform
from shortcutminer.intervention.protocol import ChannelRatioTransform


def spec():
    return {'candidate_id':'C003','status':'measurable','kind':'ratio',
            'descriptor_key':'rgb_mean_red','denominator_key':'rgb_mean_blue'}


def inputs(tmp_path, symmetric=False):
    rows=[]
    for i in range(4):
        rgb=np.full((20,20,3),[160+i*5,90,40],dtype=np.uint8)
        if symmetric:
            rgb[:,:10]=[190,80,30];rgb[:,10:]=[30,80,190]
        Image.fromarray(rgb).save(tmp_path/f'{i}.png')
        rows.append({'path':f'{i}.png','split':'train','y':i%2})
    rows.append({'path':'heldout-must-not-be-opened.png','split':'id','y':0})
    with (tmp_path/'metadata.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return SimpleNamespace(model_metadata_file='metadata.csv',split_column='split',
        train_split_values=['train'],sample_path_column='path',max_train_samples=4)


def test_reject_luminance_and_uniform_brightness_for_chromaticity():
    contract=binding_contract(spec()); key='ratio:rgb_mean_red/rgb_mean_blue'
    with pytest.raises(ValueError,match='Target-operation mismatch'):
        validate_operation(contract,'C003',key,{'type':'luminance'})
    with pytest.raises(ValueError,match='Equal channel scales'):
        validate_operation(contract,'C003',key,{'type':'global_color','mode':'channel_scale','channel_scale':[2,2,2]})
    validate_operation(contract,'C003',key,{'type':'global_color','mode':'swap_red_blue'})


def test_scalar_saturation_does_not_advertise_channel_ratio_calibration():
    from shortcutminer.experiment.autonomous_construction import ScalarChromaticityProposal
    schema=ScalarChromaticityProposal.model_json_schema()
    assert schema['properties']['channel_ratio_calibration']['type']=='null'
    assert schema['properties']['luminance_calibration']['type']=='null'
    contract=binding_contract({'candidate_id':'C003','status':'measurable','kind':'descriptor',
        'descriptor_key':'saturation_mean','denominator_key':None})
    assert contract['eligible_operation_types']==['global_color']
    with pytest.raises(ValueError,match='Target-operation mismatch'):
        validate_operation(contract,'C003','saturation_mean',{'type':'channel_ratio',
            'numerator_channel':'red','denominator_channel':'blue'})
    validate_operation(contract,'C003','saturation_mean',{'type':'global_color','mode':'desaturate'})


def test_pixel_changes_are_not_a_substitute_for_target_change(tmp_path):
    resources=inputs(tmp_path,symmetric=True)
    records,sheet=preview_construction(GlobalColorTransform(type='global_color',mode='swap_red_blue'),
        resources,tmp_path,tmp_path/'preview')
    assert all(r['changed_pixel_fraction']>0 for r in records)
    check=measure_preview_target(binding_contract(spec()),records,tmp_path,sheet.parent)
    assert check['valid_pair_count']==4 and check['changed_target_pair_count']==0
    assert check['status']=='no_measured_target_change'


class Client:
    def __init__(self):self.quant_calls=0;self.feedback=None
    def invoke_agent_validated(self,agent,payload,*,validator,**kwargs):
        if agent=='quantitative':
            self.quant_calls+=1
            self.feedback=payload['feedback']
            wrong=self.quant_calls==1
            result={'status':'selected','candidate_id':'C003','descriptor_key':'ratio:rgb_mean_red/rgb_mean_blue',
                'transformation':None if wrong else {'type':'global_color','mode':'swap_red_blue'},
                'luminance_calibration':{'mode':'normalize_mean','strategy':'pooled_median'} if wrong else None,
                'rationale':'Select from supported operations using training evidence only.',
                'intended_invariants':['Same image dimensions.'],'limitations':['Task semantics remain unverified.']}
        else:
            result={'decision':'accept','rationale':'Authorize a scoped diagnostic only.','limitations':['Not a causal claim.']}
        validator(result);return result


class Vision:
    def invoke_multimodal(self,*,payload_for_hash,validator,**kwargs):
        result={'status':'unverified','reviewed_pair_ids':[p['pair_id'] for p in payload_for_hash['training_pairs']],
            'cue_change_observation':'Actual pairs differ.','task_content_observation':'Semantic preservation unverified.',
            'artifacts_observation':'No shape change.','limitations':['Fixture review is not scientific evidence.']}
        validator(result);return result


@pytest.mark.parametrize('opt_in',[False,True])
def test_bounded_revision_and_legacy_default(tmp_path,opt_in):
    resources=inputs(tmp_path);client=Client()
    kwargs={'measurement_binding':spec()} if opt_in else {}
    result=select_autonomous_construction(client,Vision(),run_id='fixture',
        candidates=[{'candidate_id':'C003'}],observations=[],descriptor_keys=['ratio:rgb_mean_red/rgb_mean_blue'],
        resources=resources,root=tmp_path,receipt_path=tmp_path/'receipts/construction.json',
        task_description='Opaque image classification.',max_revisions=1,**kwargs)
    assert result['status']=='selected' and result['heldout_seen'] is False
    assert set(result['training_source_hashes'])=={f'{i}.png' for i in range(4)}
    if opt_in:
        assert client.quant_calls==2
        assert 'measurement_alignment_error' in client.feedback[0]
        assert result['transformation']['type']=='global_color'
        assert result['measurement_alignment']['changed_target_pair_count']==4
        assert result['semantic_preservation']=='unverified'
    else:
        assert client.quant_calls==1 and result['transformation']['type']=='luminance'
        assert 'measurement_alignment' not in result
    calls=client.quant_calls
    resumed=select_autonomous_construction(client,Vision(),run_id='fixture',
        candidates=[{'candidate_id':'C003'}],observations=[],descriptor_keys=['ratio:rgb_mean_red/rgb_mean_blue'],
        resources=resources,root=tmp_path,receipt_path=tmp_path/'receipts/construction.json',
        task_description='Opaque image classification.',max_revisions=1,**kwargs)
    assert resumed==result and client.quant_calls==calls


@pytest.mark.parametrize('mode',['normalize_mean','reflect_log_ratio'])
def test_training_calibrated_ratio_changes_target_without_geometry_change(tmp_path,mode):
    resources=inputs(tmp_path)
    colors=[[32,60,32],[64,60,16],[128,60,8],[64,60,16]]
    for i,color in enumerate(colors): Image.fromarray(np.full((20,20,3),color,np.uint8)).save(tmp_path/f'{i}.png')
    calibrated=calibrate_channel_ratio(ChannelRatioCalibrationChoice(mode=mode,strategy='pooled_median'),
        binding_contract(spec()),resources,tmp_path)
    operation=ChannelRatioTransform.model_validate(calibrated['transformation'])
    assert operation.center_ratio==pytest.approx(4) and calibrated['sample_count']==4
    for i,expected in enumerate([4,4,4,4] if mode=='normalize_mean' else [16,4,1,4]):
        before=np.array(Image.open(tmp_path/f'{i}.png'));after=transform(before,operation)
        assert after.shape==before.shape and np.array_equal(after[:,:,1],before[:,:,1])
        assert after[:,:,0].mean()/after[:,:,2].mean()==pytest.approx(expected)
    assert all('heldout' not in ref for ref in calibrated['image_hashes'])


def test_zero_channel_is_not_replaced_by_an_arbitrary_epsilon(tmp_path):
    resources=inputs(tmp_path)
    Image.fromarray(np.full((20,20,3),[20,40,0],np.uint8)).save(tmp_path/'0.png')
    with pytest.raises(ValueError,match='zero means'):
        calibrate_channel_ratio(ChannelRatioCalibrationChoice(mode='normalize_mean',strategy='pooled_median'),
            binding_contract(spec()),resources,tmp_path)


def test_agent_can_choose_fitted_ratio_recipe(tmp_path):
    resources=inputs(tmp_path)
    class FittedClient(Client):
        def invoke_agent_validated(self,agent,payload,*,validator,**kwargs):
            if agent!='quantitative':return super().invoke_agent_validated(agent,payload,validator=validator,**kwargs)
            result={'status':'selected','candidate_id':'C003','descriptor_key':'ratio:rgb_mean_red/rgb_mean_blue',
                'channel_ratio_calibration':{'mode':'normalize_mean','strategy':'pooled_median'},
                'rationale':'Training-fitted channel ratio diagnostic.','intended_invariants':['Geometry'],
                'limitations':['Subject color and task semantics may change.']}
            validator(result);return result
    result=select_autonomous_construction(FittedClient(),Vision(),run_id='fixture',candidates=[{'candidate_id':'C003'}],
        observations=[],descriptor_keys=['ratio:rgb_mean_red/rgb_mean_blue'],resources=resources,root=tmp_path,
        receipt_path=tmp_path/'receipt/construction.json',task_description='Opaque classification.',
        measurement_binding=spec(),max_revisions=0)
    assert result['status']=='selected' and result['transformation']['type']=='channel_ratio'
    assert result['calibration']['sample_count']==4 and result['measurement_alignment']['changed_target_pair_count']>0


def test_agent_cannot_supply_a_guessed_ratio_center():
    from shortcutminer.experiment.autonomous_construction import AlignedConstructionProposal
    with pytest.raises(ValueError,match='not guessed ratio centers'):
        AlignedConstructionProposal.model_validate({'status':'selected','candidate_id':'C003',
            'descriptor_key':'ratio:rgb_mean_red/rgb_mean_blue','transformation':{'type':'channel_ratio',
            'mode':'normalize_mean','numerator_channel':'red','denominator_channel':'blue','center_ratio':1},
            'rationale':'Guess a center.','intended_invariants':['Geometry'],'limitations':['Unknown.']})
