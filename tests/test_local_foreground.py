"""Local model provenance and paired statistics boundaries, without provider calls."""
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from pydantic import ValidationError
from shortcutminer.protocol.segmentation import LocalModelFiles,LocalForegroundSpec
from shortcutminer.intervention.local_foreground import verify_model_files
from shortcutminer.intervention.segmentation import SegmentationError
from shortcutminer.intervention.protocol import BackgroundReplaceTransform
from shortcutminer.experiment.paired_evaluation import evaluate_paired_members
from shortcutminer.experiment.execution_contract import _authorize_transformation,UnauthorizedScientificField
from test_quantitative_finalizer import _request

def local_files(tmp_path):
    files={}
    for name in ['config.json','preprocessor_config.json','model.safetensors']:
        path=tmp_path/name;path.write_bytes(name.encode());files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return LocalModelFiles(directory=str(tmp_path),files_sha256=files)

def test_changed_local_checkpoint_is_rejected(tmp_path):
    frozen=local_files(tmp_path)
    assert verify_model_files(frozen)==tmp_path.resolve()
    (tmp_path/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(SegmentationError,match='hash changed'):verify_model_files(frozen)

def test_missing_processor_identity_and_remote_paths_rejected(tmp_path):
    frozen=local_files(tmp_path)
    bad=frozen.model_copy(update={'files_sha256':{'model.safetensors':frozen.files_sha256['model.safetensors']}})
    with pytest.raises(SegmentationError,match='every required'):verify_model_files(bad)
    with pytest.raises(SegmentationError,match='absolute'):verify_model_files(frozen.model_copy(update={'directory':'facebook/sam-vit-base'}))

def test_model_path_or_parameter_cannot_be_changed_by_provider(tmp_path):
    files=local_files(tmp_path)
    recipe=BackgroundReplaceTransform(type='background_replace',local_foreground=LocalForegroundSpec(sam=files,grounding_dino=files))
    resources=_request(tmp_path).execution_resources.model_copy(update={'background_ood_specs':{'site':recipe}})
    assert _authorize_transformation(recipe,resources,tmp_path)==recipe
    changed=recipe.model_copy(update={'fill_color':[0,0,0]})
    with pytest.raises(UnauthorizedScientificField):_authorize_transformation(changed,resources,tmp_path)
    with pytest.raises(ValidationError):BackgroundReplaceTransform(type='background_replace',mask_column='mask',local_foreground=recipe.local_foreground)

def test_paired_members_preserve_effect_direction_and_every_seed():
    class Model:
        classes_=np.array(['0','1'])
        def member_logits(self,paths):
            if paths==['before']: rows=np.array([[5,0],[0,5],[5,0],[0,5]])
            else: rows=np.array([[0,5],[0,5],[5,0],[5,0]])
            return {f'{a}-seed{s}':rows for a in ['resnet18','resnet50'] for s in [42,43,44]}
    parameters=SimpleNamespace(confidence_level=.95,bootstrap_iterations=50,seed=42)
    result=evaluate_paired_members(Model(),SimpleNamespace(model_hash='x'),['before'],['after'],list('abcd'),list('0101'),parameters)
    assert len(result['profiles'])==6
    for profile in result['profiles']:
        assert profile['accuracy_drop']==.5 and profile['prediction_flip_rate']==.5
    assert all(len(rows)==4 for rows in result['predictions'].values())
    reverse=evaluate_paired_members(Model(),SimpleNamespace(model_hash='x'),['after'],['before'],list('abcd'),list('0101'),parameters)
    assert all(p['accuracy_drop']==-.5 for p in reverse['profiles'])
