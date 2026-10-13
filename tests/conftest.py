"""Unit tests never load a developer's local provider secrets."""
import pytest


@pytest.fixture(autouse=True)
def disable_local_provider_secret_loading(monkeypatch):
    monkeypatch.setenv("SHORTCUTMINER_DISABLE_FORMAL_ENV", "1")
