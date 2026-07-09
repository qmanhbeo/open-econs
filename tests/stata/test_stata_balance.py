"""Stata parity tests for Covariate Balance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import run_do, stata_available, DO_DIR

pytestmark = pytest.mark.skipif(
    not stata_available(), reason="StataMP not found"
)


def _stata(label: str) -> dict[str, float]:
    run_do(label)
    df = pd.read_stata(DO_DIR / f"{label}.dta")
    return dict(zip(df["name"], df["value"]))


class TestBalanceBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_ols):
        self.s = _stata("balance_basic")
        df = df_ols.copy()
        df["treat"] = (df["province"] == "north").astype(float)
        self.oe_r = oe.balance(df, treatment="treat", covariates=["x1", "x2"])

    def test_diff_x1(self):
        assert abs(self.oe_r.loc["x1", "mean_diff"] - self.s["diff_x1"]) < 1e-6

    def test_diff_x2(self):
        assert abs(self.oe_r.loc["x2", "mean_diff"] - self.s["diff_x2"]) < 1e-6
