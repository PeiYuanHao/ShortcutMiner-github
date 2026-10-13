import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from shortcutminer.experiment.spatial_construction_quality import select_spatial_construction
from shortcutminer.experiment.ood_construction_quality import construction_quality
from shortcutminer.intervention.protocol import SpatialTransform


def setup(tmp_path):
    rgb = np.full((32, 32, 3), 60, dtype=np.uint8)
    rgb[20:28, 16:30] = 230
    Image.fromarray(rgb).save(tmp_path/'train.png')
    pd.DataFrame([{'path':'train.png','split':'train'},
                  {'path':'heldout-must-not-open.png','split':'id'}]).to_csv(tmp_path/'metadata.csv',index=False)
    return SimpleNamespace(model_metadata_file='metadata.csv', split_column='split',
                           train_split_values=['train'], sample_path_column='path')


def response(custom=False):
    box={'x0':.5,'y0':.625,'x1':.9375,'y1':.875}
    return {'status':'selected','candidate_id':'C001','descriptor_key':'edge', 'target_box':box,
            'transformation': {'type':'spatial_mask','region':'custom' if custom else 'bottom_right',
                               'custom_box':box if custom else None, 'fill_color':[0,0,0]},
            'rationale':'Training-localized overlay test.', 'intended_invariants':['Outside pixels identical.'],
            'semantic_risks':['Task content can be removed.'], 'processing_controls':['No-op PNG.'],
            'limitations':['Target semantics unverified.']}


def select(tmp_path, client, **kwargs):
    class WithLocalization:
        def invoke_agent(self,name,payload,**options):
            if payload['mode']=='autonomous_visual_localization':
                assert 'tool_coordinates' not in payload
                r=response(custom=True)
                return {k:r[k] for k in ['candidate_id','descriptor_key','target_box','rationale']}
            return client.invoke_agent(name,payload,**options)
    return select_spatial_construction(WithLocalization(), candidates=[{'candidate_id':'C001'}],observations=[],
        descriptor_keys=['edge'],resources=setup(tmp_path),root=tmp_path,
        receipt_path=tmp_path/'receipt.json',prompt='Training-only protocol.',**kwargs)


def test_undercoverage_repaired_by_agent_before_training_and_resume(tmp_path):
    class Client:
        calls=0
        def invoke_agent(self,name,payload,**kwargs):
            assert name=='quantitative'
            if self.calls:
                quality=payload['construction_quality_feedback'][-1]['construction_quality']
                assert quality['status']=='failed'
                assert quality['samples'][0]['uncovered_target_pixels']>0
            self.calls+=1
            return response(custom=self.calls>1)
    client=Client(); proposal=select(tmp_path,client)
    assert proposal.transformation.region=='custom' and client.calls==2
    saved=json.loads((tmp_path/'receipt.json').read_text())
    assert len(saved['attempts'])==2 and saved['attempts'][-1]['construction_quality']['status']=='passed'
    assert not saved['heldout_seen'] and not saved['truth_seen']
    select(tmp_path,client); assert client.calls==2
    saved['proposal']['rationale']='tampered'
    (tmp_path/'receipt.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='record changed'): select(tmp_path,client)


def test_budget_exhaustion_defers_without_fallback(tmp_path):
    class Client:
        calls=0
        def invoke_agent(self,*args,**kwargs): self.calls+=1; return response()
    client=Client()
    with pytest.raises(ValueError,match='deferred'): select(tmp_path,client)
    assert client.calls==3
    assert json.loads((tmp_path/'receipt.json').read_text())['status']=='deferred_construction_quality'


def test_agent_cannot_shrink_target_to_fake_coverage(tmp_path):
    class Client:
        calls=0
        def invoke_agent(self,*args,**kwargs):
            result=response(); self.calls+=1
            if self.calls>1: result['target_box']={'x0':.75,'y0':.75,'x1':1,'y1':1}
            return result
    with pytest.raises(ValueError,match='shrink/change'): select(tmp_path,Client())


def test_optin_geometry_mismatch_is_rejected_and_bounded_retry_preserves_localization(tmp_path):
    class Client:
        calls=0
        def invoke_agent(self,name,payload,**kwargs):
            self.calls+=1
            result=response(custom=True)
            if self.calls==1:
                result['target_box']={'x0':.75,'y0':.75,'x1':1,'y1':1}
            else:
                assert payload['construction_quality_feedback'][-1]['construction_quality']['status']=='failed'
            return result
    client=Client(); selected=select(tmp_path,client,allow_exterior=True)
    assert client.calls==2 and selected.target_box.model_dump()==response(custom=True)['target_box']
    saved=json.loads((tmp_path/'receipt.json').read_text())
    assert saved['attempts'][0]['construction_quality']['status']=='failed'
    assert saved['attempts'][1]['construction_quality']['status']=='passed'


def test_spatial_dispatch_cannot_silently_pass_without_localization(tmp_path):
    resources=setup(tmp_path)
    operation=SpatialTransform.model_validate(response()['transformation'])
    assert construction_quality(operation,resources,tmp_path)['status']=='failed'


def test_training_path_escape_is_rejected(tmp_path):
    resources=setup(tmp_path)
    pd.DataFrame([{'path':'../outside.png','split':'train'}]).to_csv(tmp_path/'metadata.csv',index=False)
    Image.new('RGB',(32,32)).save(tmp_path.parent/'outside.png')
    with pytest.raises(ValueError): construction_quality(SpatialTransform.model_validate(response()['transformation']),resources,tmp_path)


@pytest.mark.parametrize('size,band', [(150,30),(97,13)])
def test_public_exact_border_residual_requires_complete_coverage(size, band):
    from shortcutminer.experiment.spatial_construction_quality import spatial_quality, uniform_edge_regions
    from shortcutminer.intervention.protocol import NormalizedBox
    rng=np.random.default_rng(12)
    rgb=rng.integers(0,256,(size,size,3),dtype=np.uint8)
    rgb[:band]=[221,17,91]
    target=NormalizedBox(x0=0,y0=0,x1=1,y1=(band-2)/size)
    partial=SpatialTransform(type='spatial_replace',region='custom',custom_box=target,fill='constant')
    samples=[('public-training.png','source',rgb)]
    assert uniform_edge_regions(rgb)[0]['pixel_box']==[0,0,size,band]
    assert spatial_quality(partial,samples,target_box=target)['status']=='passed'
    checked=spatial_quality(partial,samples,target_box=target,check_edge_residual=True)
    assert checked['status']=='failed'
    assert checked['samples'][0]['residual_edge_regions'][0]['uncovered_pixels']==2*size
    full=partial.model_copy(update={'custom_box':NormalizedBox(x0=0,y0=0,x1=1,y1=band/size)})
    assert spatial_quality(full,samples,target_box=target,check_edge_residual=True)['status']=='passed'


def test_exterior_frame_reversal_preserves_content_and_has_guarded_area_exception():
    from shortcutminer.experiment.spatial_construction_quality import spatial_quality
    from shortcutminer.intervention.protocol import NormalizedBox
    from shortcutminer.intervention.engine import _transform_basic
    rgb=np.full((40,40,3),[220,20,70],dtype=np.uint8)
    rng=np.random.default_rng(27)
    rgb[10:30,8:32]=rng.integers(0,256,(20,24,3),dtype=np.uint8)
    inner=NormalizedBox(x0=.2,y0=.25,x1=.8,y1=.75)
    spec=SpatialTransform(type='spatial_replace',region='custom_exterior',custom_box=inner,fill='color_complement')
    changed=_transform_basic(rgb,spec)
    assert np.array_equal(changed[10:30,8:32],rgb[10:30,8:32])
    assert changed[0,0].tolist()==[35,235,185]
    assert np.array_equal(_transform_basic(changed,spec),rgb)
    samples=[('train.png','sha',rgb)]
    assert spatial_quality(spec,samples,target_box=inner,target_geometry='exterior')['status']=='failed'
    quality=spatial_quality(spec,samples,target_box=inner,target_geometry='exterior',allow_exterior=True,check_edge_residual=True)
    assert quality['status']=='passed'
    assert quality['samples'][0]['mask_fraction']>.5
    assert quality['samples'][0]['measured_border_area_exception']
    # An enlarged intervention intruding into content cannot use this exception.
    smaller=NormalizedBox(x0=.25,y0=.3,x1=.75,y1=.7)
    unsafe=spec.model_copy(update={'custom_box':smaller})
    assert spatial_quality(unsafe,samples,target_box=inner,target_geometry='exterior',allow_exterior=True)['status']=='failed'
