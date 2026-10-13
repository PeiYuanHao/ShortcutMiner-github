import numpy as np
import pytest
from shortcutminer.experiment.waterbirds_background_validation import paired_metrics, protect_mask


def test_soft_annotation_support_is_conservatively_protected():
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[1, 1] = 51
    mask = protect_mask(image, image.shape, margin=0)
    assert mask.sum() == 1 and mask[1, 1]
    assert protect_mask(image, image.shape, margin=1).all()


def test_empty_colored_or_misaligned_masks_rejected():
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Empty"):
        protect_mask(mask, mask.shape)
    mask[1, 1, 0] = 255
    with pytest.raises(ValueError, match="grayscale"):
        protect_mask(mask, mask.shape)
    with pytest.raises(ValueError, match="dimensions"):
        protect_mask(mask, (4, 3, 3))


def test_paired_statistics_retain_improvements_and_failures():
    result = paired_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 1, 1], seed=42)
    assert result["prediction_flip_rate"] == 0.5
    assert result["accuracy_delta"] == 0.0
    assert result["correct_to_wrong"] == result["wrong_to_correct"] == 1
    assert result["accuracy_delta_ci"] == [0.0, 0.0]
    assert result["groups"]["0:0"]["accuracy_delta"] == -1
    assert result["groups"]["1:0"]["accuracy_delta"] == 1


def test_unaligned_pairs_rejected():
    with pytest.raises(ValueError, match="aligned"):
        paired_metrics([0, 1], [0, 1], [0], [0, 1], seed=42)
