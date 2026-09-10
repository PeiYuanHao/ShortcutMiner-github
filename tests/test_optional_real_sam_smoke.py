import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("SHORTCUTMINER_REAL_SAM_TEST") != "1"
    or not os.environ.get("SHORTCUTMINER_SAM_CHECKPOINT"),
    reason="real SAM smoke requires explicit opt-in and an approved local checkpoint",
)
def test_optional_real_sam_configuration_is_local_only():
    checkpoint = Path(os.environ["SHORTCUTMINER_SAM_CHECKPOINT"])
    assert checkpoint.is_absolute()
    assert checkpoint.is_file()
