from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shortcutminer.experiment.adaptive_screening import prepare_probe
from shortcutminer.intervention.protocol import SpatialTransform, FactorTransform, GlobalColorTransform
from shortcutminer.protocol.model_reliance import AdaptivePairedScreeningOptions
from shortcutminer.protocol.quantitative import QuantitativeExecutionResources


@pytest.mark.parametrize('operation',['spatial','exterior','brightness','channel_scale','desaturate','luminance','rotation','frequency','blur','noise','compression','resolution'])
def test_generated_spatial_probe_keeps_pixels_and_reserves_confirmation(tmp_path,operation):
    spec=SpatialTransform(type='spatial_mask',region='custom',custom_box={'x0':.5,'y0':.5,'x1':1,'y1':1},fill='constant') if operation=='spatial' else FactorTransform(type='brightness',factor=.5)
    if operation=='exterior':
        spec=SpatialTransform(type='spatial_replace',region='custom_exterior',custom_box={'x0':.25,'y0':.25,'x1':.75,'y1':.75},fill='color_complement')
    if operation=='channel_scale':
        spec=GlobalColorTransform(type='global_color',mode='channel_scale',channel_scale=[.5,1,1])
    if operation=='desaturate':
        spec=GlobalColorTransform(type='global_color',mode='desaturate',desaturation_factor=0)
    from shortcutminer.intervention.protocol import LuminanceTransform,RotationTransform,FrequencyTransform,BlurTransform,NoiseTransform,CompressionTransform,ResolutionTransform
    additional={
        'luminance':LuminanceTransform(type='luminance',mode='normalize_mean',target_mean=110),
        'rotation':RotationTransform(type='rotation',angle_degrees=10),
        'frequency':FrequencyTransform(type='frequency_filter',mode='low_pass',cutoff=.2),
        'blur':BlurTransform(type='blur',radius=1),
        'noise':NoiseTransform(type='noise',std=10,seed=3),
        'compression':CompressionTransform(type='compression',quality=40),
        'resolution':ResolutionTransform(type='resolution',width=2,height=2),
    }
    spec=additional.get(operation,spec)
    rows=[]
    for i in range(128):
        rgb=np.full((8,8,3),60,dtype=np.uint8)
        rgb[0,0]=[i,80,90]
        if operation=='exterior': rgb[2,2]=[i,80,90]
        rgb[4:,4:]=[255,0,0] if i%2 else [0,0,255]
        Image.fromarray(rgb).save(tmp_path/f'{i}.png')
        rows.append({'id':str(i),'path':f'{i}.png','y':str(i%2),'split':'id'})
    options=AdaptivePairedScreeningOptions(minimum_effect=.02,family_alpha=.05,candidate_budget=1,
        pair_checkpoints=[32,64],probe_fraction=.5,sensitivity_effects=[.01,.02,.05])
    params=SimpleNamespace(adaptive_screening=options,screening_transformation=spec,split_column='split',
        screening_split_values=['id'],sample_id_column='id',sample_path_column='path',label_column='y',seed=42,
        model_dump=lambda **kw:{'transformation':spec.model_dump(mode='json'),'options':options.model_dump(mode='json')})
    access=SimpleNamespace(dataset_file=lambda ref:tmp_path/ref,context=SimpleNamespace(dataset_root=tmp_path))
    class Model:
        classes_=np.array(['0','1'])
        _screening_validation_labels=['0','1']
        def _infer(self,fitted,paths):
            values=[]
            for path in paths:
                a=np.array(Image.open(path)).astype(float)
                d=(a[4:,4:,0]-a[4:,4:,2]).mean()
                if operation in {'brightness','channel_scale','desaturate'}: d=a[4:,4:,0].mean()-200
                values.append([-d,d])
            return np.array(values)
    model=Model()
    reserved,callback=prepare_probe(params,access,pd.DataFrame(rows),model,tmp_path/'probe')
    assert len(reserved)==64 and len(set(r['id'] for r in rows)-set(reserved))==64
    expanded=callback(None,{'weights_sha256':'fixture','validation_accuracy':1.0})
    if operation not in additional:
        assert expanded and model.adaptive_report['decision']=='expand_models'
    else:
        assert expanded == (model.adaptive_report['decision']=='expand_models')
        assert model.adaptive_report['looks']  # Actual paired predictions, even when effect is null.

    assert model.adaptive_report['sampling_unit']==('exact_content_outside_intervention_region' if operation in {'spatial','exterior'} else 'exact_source_pixels_not_verified_physical_instances')
    for evidence in model.adaptive_report['pixel_evidence']:
        before=np.array(Image.open(tmp_path/evidence['source_reference']))
        after=np.array(Image.open(tmp_path/'probe'/evidence['changed_reference']))
        if operation=='spatial':
            assert np.array_equal(before[:4],after[:4])
            assert np.array_equal(before[4:,:4],after[4:,:4])
        elif operation=='exterior':
            assert np.array_equal(before[2:6,2:6],after[2:6,2:6])
        else:
            from shortcutminer.intervention.engine import _transform_basic
            expected=_transform_basic(before,spec)
            if operation=='compression':
                from io import BytesIO
                buffer=BytesIO();Image.fromarray(expected).save(buffer,format='JPEG',quality=spec.quality)
                buffer.seek(0);expected=np.array(Image.open(buffer).convert('RGB'))
            assert np.array_equal(after,expected)
            assert before.shape==after.shape
    assert callback(None,{'weights_sha256':'fixture','validation_accuracy':1.0}) == expanded


def test_spatial_resources_keep_old_default_serialization():
    assert 'visual_ood_specs' not in QuantitativeExecutionResources(dataset_name='fixture',label_column='y').model_dump()
    parsed=QuantitativeExecutionResources(dataset_name='fixture',label_column='y',visual_ood_specs={'observed':{'type':'spatial_replace','region':'top_left','fill':'mean'}})
    assert isinstance(parsed.visual_ood_specs['observed'],SpatialTransform)
