from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shortcutminer.experiment.adaptive_screening import prepare_probe
from shortcutminer.intervention.protocol import GlobalColorTransform
from shortcutminer.protocol.model_reliance import AdaptivePairedScreeningOptions

def test_red_green_probe_groups_actual_digit_content_and_preserves_pixels(tmp_path):
    rows=[]
    for i in range(64):
        for color in range(2):
            rgb=np.zeros((8,8,3),dtype=np.uint8)
            rgb[2:6,2:6,color]=i+1
            reference=f'{i}-{color}.png';Image.fromarray(rgb).save(tmp_path/reference)
            rows.append({'id':f'{i}-{color}','path':reference,'label':str(color),'split':'id'})
    params=SimpleNamespace(screening_transformation=GlobalColorTransform(type='global_color',mode='swap_red_green'),
        screening_split_values=['id'],split_column='split',sample_id_column='id',sample_path_column='path',label_column='label',seed=42,
        adaptive_screening=AdaptivePairedScreeningOptions(minimum_effect=.02,family_alpha=.05,candidate_budget=3,
            pair_checkpoints=[8,16,32],probe_fraction=.5,sensitivity_effects=[.01,.02,.05]),model_dump=lambda **kwargs:{'fixture':'irm'})
    class Model:
        classes_=np.array(['0','1']);_screening_validation_labels=['0','1']
        def _infer(self, fitted, paths):
            return np.array([np.array(Image.open(p))[:,:,:2].sum((0,1)) for p in paths])
    directory=tmp_path/'model';directory.mkdir()
    reserved, callback=prepare_probe(params,SimpleNamespace(dataset_file=lambda ref:tmp_path/ref),pd.DataFrame(rows),Model(),directory)
    assert len(reserved)==64 # 32 whole digit groups, each represented by 2 colorings
    assert callback(None,{'validation_accuracy':1.,'weights_sha256':'fixture'}) is True
    report=json.loads((directory/'adaptive_screening.json').read_text())
    assert report['decision']=='expand_models' and report['looks'][0]['sample_count']==8
    assert report['looks'][0]['accuracy_drop']==1.
    assert len(report['probe_sample_ids'])==32
    for item in report['pixel_evidence']:
        original=np.array(Image.open(tmp_path/item['source_reference']))
        changed=np.array(Image.open(directory/item['changed_reference']))
        assert np.array_equal(changed,original[:,:,[1,0,2]])
        assert np.array_equal(changed.max(2),original.max(2))

def test_red_green_swap_is_exact_involution(tmp_path):
    from shortcutminer.intervention.engine import InterventionEngine
    rgb=np.zeros((14,14,3),dtype=np.uint8);rgb[2:10,3:8,0]=np.arange(40).reshape(8,5)
    original=tmp_path/'a.png';changed=tmp_path/'b.png';restored=tmp_path/'c.png'
    Image.fromarray(rgb).save(original)
    spec=GlobalColorTransform(type='global_color',mode='swap_red_green')
    engine=InterventionEngine()
    engine.apply(source_path=original,output_path=changed,transformation=spec)
    engine.apply(source_path=changed,output_path=restored,transformation=spec)
    assert np.array_equal(np.array(Image.open(restored)),rgb)
