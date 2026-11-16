"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def plans_in_tmp(tmp_path, monkeypatch):
    """Runs of the agent in tests save their plans to a temporary folder, never to out/plans."""
    monkeypatch.setenv("PLAST_PLANS_DIR", str(tmp_path / "plans"))
