"""Stata parity tests for Covariate Balance."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import read_stata


class TestBalanceBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_ols):
        self.s = read_stata("balance_basic")
        df = df_ols.copy()
        df["treat"] = (df["province"] == "north").astype(float)
        self.oe_r = oe.balance(df, treatment="treat", covariates=["x1", "x2"])

    def test_diff_x1(self):
        oe_diff = self.oe_r.loc[self.oe_r["Variable"] == "x1", "Difference"].values[0]
        # oe vs Stata t-test: slight numerical difference in implementation
        npt.assert_allclose(abs(oe_diff), abs(self.s["diff_x1"]), rtol=1e-3)

    def test_diff_x2(self):
        oe_diff = self.oe_r.loc[self.oe_r["Variable"] == "x2", "Difference"].values[0]
        npt.assert_allclose(abs(oe_diff), abs(self.s["diff_x2"]), rtol=1e-3)
