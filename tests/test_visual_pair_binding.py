import hashlib
import json
import numpy as np
import pytest
from PIL import Image
from shortcutminer.intervention.protocol import ProvidedPairTransform
from shortcutminer.intervention.engine import InterventionEngine
from shortcutminer.experiment.candidate_binding import resolve_candidate_binding
from shortcutminer.protocol.identify import Candidate
from shortcutminer.protocol.executive import ExecutiveCapabilities


def setup_pair(tmp_path, changed_outside=False):
    original = np.full((16,16,3), 90, dtype=np.uint8)
    paired = original.copy(); paired[12:,12:] = 230
    if changed_outside: paired[0,0] = 0
    first=tmp_path/'id.png'; second=tmp_path/'ood.png'
    Image.fromarray(original).save(first); Image.fromarray(paired).save(second)
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    row={'original_reference':'id.png','paired_reference':'ood.png', 'original_label':'0','paired_label':'0',
         'original_sha256':sha(first),'paired_sha256':sha(second)}
    index=tmp_path/'pairs.json'; index.write_text(json.dumps({'pairs':[row]}))
    spec=ProvidedPairTransform(type='provided_pair',pair_index_file='pairs.json',pair_index_sha256=sha(index),
                              changed_region={'x0':.5,'y0':.5,'x1':1,'y1':1})
    return first, second, spec


def test_external_pairs_preserve_pixels_and_do_not_generate_an_oracle(tmp_path):
    first,second,spec=setup_pair(tmp_path)
    output=tmp_path/'changed.png'
    quality=InterventionEngine().apply(source_path=first,output_path=output,transformation=spec,dataset_root=tmp_path)
    assert quality.same_dimensions
    assert np.array_equal(np.array(Image.open(output)),np.array(Image.open(second)))


def test_pairs_reject_changes_outside_agent_region(tmp_path):
    first,_,spec=setup_pair(tmp_path,True)
    with pytest.raises(ValueError,match='outside'):
        InterventionEngine().apply(source_path=first,output_path=tmp_path/'changed.png',transformation=spec,dataset_root=tmp_path)


def test_pairs_reject_tampered_target_and_index(tmp_path):
    first,second,spec=setup_pair(tmp_path)
    second.write_bytes(second.read_bytes()+b'x')
    with pytest.raises(ValueError,match='identity'):
        InterventionEngine().apply(source_path=first,output_path=tmp_path/'changed.png',transformation=spec,dataset_root=tmp_path)
    (tmp_path/'pairs.json').write_text('{}')
    with pytest.raises(ValueError,match='index identity'):
        InterventionEngine().apply(source_path=first,output_path=tmp_path/'changed.png',transformation=spec,dataset_root=tmp_path)


def test_visual_binding_is_explicit_and_old_behavior_is_preserved():
    candidate=Candidate(candidate_id='C001',cue={'family':'appearance','name':'localized texture'},
        description='A visual texture',observation='Visible in training images',hypothesis='May correlate with label',
        evidence_sources=['image_statistics'],priority='high',status='unverified')
    args={'available_metadata_fields':['y','id','measured_edges'],'label_column':'y','addressing_fields':['id']}
    assert resolve_candidate_binding(candidate,**args).kind=='non_metadata'
    bound=resolve_candidate_binding(candidate,**args,visual_measurement_bindings={'C001':'measured_edges'})
    assert bound.metadata_fields==['measured_edges']
    with pytest.raises(ValueError):
        resolve_candidate_binding(candidate,**args,visual_measurement_bindings={'C001':'y'})
    assert 'visual_measurement_bindings' not in ExecutiveCapabilities().model_dump()
