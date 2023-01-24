"""shared fixtures."""

import os
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    """seed before every test so anything stochastic is reproducible."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield


@pytest.fixture
def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
