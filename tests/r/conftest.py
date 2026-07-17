"""Shared fixtures for R parity tests.

Mirrors tests/stata/conftest.py: session-scoped dataset fixtures loaded
once per test session from committed CSV inputs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from .r_runner import R_INPUTS_DIR


def _load_csv(name: str) -> pd.DataFrame:
    """Load a fixed CSV from the R inputs directory."""
    path = R_INPUTS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def gmm_input() -> pd.DataFrame:
    """Shared GMM input dataset (300 obs, y/x1/x2/z1-z5/cluster/t)."""
    return _load_csv("gmm_input")


@pytest.fixture(scope="session")
def iv_input() -> pd.DataFrame:
    """Shared IV input dataset (500 obs, overidentified 2 instruments, panel)."""
    return _load_csv("iv_input")
