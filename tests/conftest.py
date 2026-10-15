from __future__ import annotations

import pytest

from sglang_maestro.deploy import DeployOptions
from sglang_maestro.target import Target

SMALL_DEPLOY = DeployOptions(sizes=(8,), capacity_samples=40, verify_samples=120)
"""Few simulated requests and one instance size: seconds instead of minutes per plan."""


@pytest.fixture(scope="session")
def dense_chat() -> Target:
    return Target.load("qwen3-32b", "h20", "chat")


@pytest.fixture(scope="session")
def oot_moe() -> Target:
    return Target.load("deepseek-v3", "template-h20-class")
