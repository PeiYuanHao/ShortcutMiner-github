"""Guard scientific boundaries of the new opt-in ESWA harness."""
from pathlib import Path
import numpy as np
from experiments.eswa_20261019.prepare_inputs import select_pool, transform
from experiments.eswa_20261019.run_one import cache_identity


def test_source_assignment_excludes_legacy_and_is_disjoint():
    records=[{'label':label,'sha256':f'{label}-{i}','pixels':None}
             for label in (3,5) for i in range(20)]
    pool=select_pool(records,(3,5),{'train':8,'val':4,'id':8},42,{'3-0','5-0'})
    assert len(pool)==20 and len({r['sha256'] for r in pool})==20
    assert not {'3-0','5-0'}&{r['sha256'] for r in pool}
    for split in ('train','val','id'):
        group=[r for r in pool if r['split']==split]
        assert sum(r['y']=='0' for r in group)==sum(r['y']=='1' for r in group)


def test_border_preserves_actual_central_content_and_target_mask():
    pixels=np.random.default_rng(4).integers(0,256,(32,32,3),dtype=np.uint8)
    first,mask=transform(pixels,'border',0,{})
    second,_=transform(pixels,'border',1,{})
    assert np.array_equal(first[4:36,4:36],pixels)
    assert np.array_equal(second[4:36,4:36],pixels)
    assert np.array_equal(np.any(first!=second,axis=2),mask)


def test_watermark_preserves_every_non_support_pixel():
    pixels=np.zeros((32,32,3),dtype=np.uint8)
    absent,_=transform(pixels,'watermark',0,{'watermark_xy':(25,29)})
    present,mask=transform(pixels,'watermark',1,{'watermark_xy':(25,29)})
    assert mask.sum()>0
    assert np.array_equal(np.any(absent!=present,axis=2),mask)


def test_shared_training_cache_uses_bytes_labels_and_recipe_not_paths(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';v=tmp_path/'val'
    a.write_bytes(b'train');b.write_bytes(b'train');v.write_bytes(b'val')
    args=([a],['0'],[v],['1'],{'seeds':[42]},{'resnet18':'hash'})
    original,_=cache_identity(*args)
    moved,_=cache_identity([b],*args[1:])
    relabeled,_=cache_identity([a],['1'],*args[2:])
    changed,_=cache_identity(*args[:4],{'seeds':[43]},args[5])
    assert original==moved
    assert len({original,relabeled,changed})==3


def test_compact_roles_preserve_scope_and_provide_exact_output_schemas():
    import json
    from experiments.eswa_20261019.compact_roles import COMMON,ROLES,supply_schema
    from shortcutminer.tooling.agent_loop import _FINAL_SCHEMAS
    for name,text in ROLES.items():
        assert len(COMMON+text)<4000
        delivered=supply_schema(name,{},COMMON+text)
        assert json.loads(delivered.split('Current output schema (caller mode is authoritative):\n')[1])==_FINAL_SCHEMAS[name].model_json_schema()
    custom={'type':'object','properties':{'decision':{'type':'string'}}}
    delivered=supply_schema('qualitative',{'response_schema':custom},COMMON)
    assert json.loads(delivered.split('Current output schema (caller mode is authoritative):\n')[1])==custom
