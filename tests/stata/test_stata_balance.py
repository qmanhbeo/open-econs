"""Stata parity tests for Covariate Balance (Welch t-tests).

Stata fixture has been regenerated with ``unequal`` (Welch) variance and
sign convention ``treated − control``, matching ``open_econs.balance()``.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from scipy import stats as _stats

from .stata_runner import read_stata


def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
    """Satterthwaite degrees of freedom for Welch's t-test."""
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    num = (s1**2 / n1 + s2**2 / n2) ** 2
    denom = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    return num / denom


class TestBalanceBasic:
    """Signed comparisons vs Stata Welch t-test fixture — no abs() masking."""

    @pytest.fixture(autouse=True)
    def _run(self, df_ols):
        self.s = read_stata("balance_basic")
        df = df_ols.copy()
        df["treat"] = (df["province"] == "north").astype(float)

        self.ref = {}
        for var in ["x1", "x2"]:
            treated = df.loc[df["treat"] == 1, var].values
            control = df.loc[df["treat"] == 0, var].values

            diff = treated.mean() - control.mean()
            t_stat, p_val = _stats.ttest_ind(treated, control, equal_var=False)
            self.ref[var] = {
                "diff": diff,
                "t": t_stat,
                "p": p_val,
                "df": _welch_df(treated, control),
            }

    # --- x1 ---

    def test_diff_x1(self):
        npt.assert_allclose(self.ref["x1"]["diff"], self.s["diff_x1"], rtol=1e-6)

    def test_t_x1(self):
        npt.assert_allclose(self.ref["x1"]["t"], self.s["t_x1"], rtol=1e-6)

    def test_df_x1(self):
        npt.assert_allclose(self.ref["x1"]["df"], self.s["df_x1"], rtol=1e-6)

    def test_p_x1(self):
        npt.assert_allclose(self.ref["x1"]["p"], self.s["p_x1"], rtol=1e-6)

    # --- x2 ---

    def test_diff_x2(self):
        npt.assert_allclose(self.ref["x2"]["diff"], self.s["diff_x2"], rtol=1e-6)

    def test_t_x2(self):
        npt.assert_allclose(self.ref["x2"]["t"], self.s["t_x2"], rtol=1e-6)

    def test_df_x2(self):
        npt.assert_allclose(self.ref["x2"]["df"], self.s["df_x2"], rtol=1e-6)

    def test_p_x2(self):
        npt.assert_allclose(self.ref["x2"]["p"], self.s["p_x2"], rtol=1e-6)
