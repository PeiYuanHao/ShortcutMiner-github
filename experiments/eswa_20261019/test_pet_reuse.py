from copy import deepcopy
import pytest
from experiments.eswa_20261019.prepare_pet import validate_recipe


def test_historical_member_can_be_reused_only_with_same_recipe_and_existing_axes():
    old={'architectures':['resnet18','resnet50'],'seeds':[42,43,44],'learning_rate':0.001,
        'augmentation':'spatial_preserving','epochs':100}
    selected={**old,'architectures':['resnet18'],'seeds':[42]}
    validate_recipe(old,selected)
    for field,value in [('learning_rate',0.01),('augmentation','natural_image'),('seeds',[99]),('architectures',['other'])]:
        changed=deepcopy(selected);changed[field]=value
        with pytest.raises(ValueError):validate_recipe(old,changed)
