"""Shared fixtures for Stata parity tests.

Dual-mode behaviour:
  - If StataMP is installed locally, .do files are re-run and .dta files
    regenerated before each test.
  - If StataMP is absent (e.g. CI), the committed .dta fixtures are used.
  - A drift check fails the test if a .do file is newer than its .dta,
    catching stale fixtures when someone edits a .do but forgets to regenerate.

Fixtures are fixed CSV datasets committed to tests/stata/fixtures/.
Both Python (open-econs) and Stata operate on the exact same data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from .stata_runner import FIXTURES_DIR


def _load_csv(name: str) -> pd.DataFrame:
    """Load a fixed CSV from the fixtures directory."""
    path = FIXTURES_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def df_ols() -> pd.DataFrame:
    return _load_csv("df_ols")


@pytest.fixture(scope="session")
def df_panel() -> pd.DataFrame:
    return _load_csv("df_panel")


@pytest.fixture(scope="session")
def df_iv() -> pd.DataFrame:
    return _load_csv("df_iv")


@pytest.fixture(scope="session")
def df_logit() -> pd.DataFrame:
    return _load_csv("df_logit")


@pytest.fixture(scope="session")
def df_did() -> pd.DataFrame:
    return _load_csv("df_did")


@pytest.fixture(scope="session")
def df_oaxaca() -> pd.DataFrame:
    return _load_csv("df_oaxaca")


@pytest.fixture(scope="session")
def df_rdd() -> pd.DataFrame:
    return _load_csv("df_rdd")
