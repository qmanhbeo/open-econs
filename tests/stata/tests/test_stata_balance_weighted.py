"""Stata parity tests for weighted covariate balance.

Weight convention (pstest-style):
  treated = PSM weight (1), control ~ Unif[0.5, 2.5], seed 20240711.

Every reference value was produced by real Stata commands inside
``balance_weighted.do``:
  - ``summarize [iw=w]`` for weighted means and variances
  - ``regress [iw=w]`` for weighted OLS t-stats and p-values
  - scalar arithmetic from those Stata-produced numbers for SMD and VR

The SMD is stored as a raw ratio (not pstest's ×100 %-bias).  The test
compares raw computed values to Stata, not the rounded DataFrame output.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from open_econs.models.causal.balance import (
    _wls_t,
    _wmean,
    _wvar_iw,
)

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

S = read_stata("balance_weighted")


class TestWeightedStats:
    """Weighted means, SMD, VR, and WLS t-test vs Stata [iw=w] fixture."""

    @pytest.fixture(autouse=True)
    def _run(self, df_balance_weighted):
        df = df_balance_weighted.copy()
        w = df["w"].values

        self.ref = {}
        for var in ["x1", "x2"]:
            is_treat = df["t"].values == 1.0
            t_vals = df.loc[is_treat, var].values
            c_vals = df.loc[~is_treat, var].values
            t_w = w[is_treat]
            c_w = w[~is_treat]

            t_mean = _wmean(t_vals, t_w)
            c_mean = _wmean(c_vals, c_w)
            diff = t_mean - c_mean
            t_uvar = np.var(t_vals, ddof=1)
            c_uvar = np.var(c_vals, ddof=1)
            smd = diff / np.sqrt((t_uvar + c_uvar) / 2)
            vr = _wvar_iw(t_vals, t_w) / _wvar_iw(c_vals, c_w)

            not_nan = ~df[var].isna().values
            tstat, pval = _wls_t(
                df["t"].values[not_nan].astype(float),
                df[var].values[not_nan],
                w[not_nan],
            )

            self.ref[var] = {
                "diff": diff,
                "smd": smd,
                "vr": vr,
                "t": tstat,
                "p": pval,
            }

    # --- x1 ---
    def test_diff_x1(self):
        npt.assert_allclose(self.ref["x1"]["diff"], S["diff_x1"], rtol=1e-6)

    def test_smd_x1(self):
        npt.assert_allclose(self.ref["x1"]["smd"], S["smd_x1"], rtol=1e-6)

    def test_vr_x1(self):
        npt.assert_allclose(self.ref["x1"]["vr"], S["vr_x1"], rtol=1e-6)

    def test_t_x1(self):
        npt.assert_allclose(self.ref["x1"]["t"], S["tstat_x1"], rtol=1e-6)

    def test_p_x1(self):
        npt.assert_allclose(self.ref["x1"]["p"], S["pval_x1"], atol=1e-12)

    # --- x2 ---
    def test_diff_x2(self):
        npt.assert_allclose(self.ref["x2"]["diff"], S["diff_x2"], rtol=1e-6)

    def test_smd_x2(self):
        npt.assert_allclose(self.ref["x2"]["smd"], S["smd_x2"], rtol=1e-6)

    def test_vr_x2(self):
        npt.assert_allclose(self.ref["x2"]["vr"], S["vr_x2"], rtol=1e-6)

    def test_t_x2(self):
        npt.assert_allclose(self.ref["x2"]["t"], S["tstat_x2"], rtol=1e-6)

    def test_p_x2(self):
        npt.assert_allclose(self.ref["x2"]["p"], S["pval_x2"], atol=1e-12)


class TestBalanceFunctionWeighted:
    """The balance() wrapper produces the right columns and correct values."""

    def test_column_names(self, df_balance_weighted):
        from open_econs.models.causal.balance import balance

        result = balance(
            df_balance_weighted,
            treatment="t",
            covariates=["x1", "x2"],
            weights="w",
        )
        expected = [
            "Variable",
            "Treated Mean",
            "Control Mean",
            "Difference",
            "Treated Std",
            "Control Std",
            "SMD",
            "Variance Ratio",
            "t-statistic",
            "P>|t|",
        ]
        assert result.columns.tolist() == expected

    def test_variable_order(self, df_balance_weighted):
        from open_econs.models.causal.balance import balance

        result = balance(
            df_balance_weighted,
            treatment="t",
            covariates=["x1", "x2"],
            weights="w",
        )
        assert list(result["Variable"]) == ["x1", "x2"]

    def test_weighted_values_rounded_correctly(self, df_balance_weighted):
        """Rounding to 4dp means each value is within ±0.0001 of the raw Stata ref."""
        from open_econs.models.causal.balance import balance

        result = balance(
            df_balance_weighted,
            treatment="t",
            covariates=["x1", "x2"],
            weights="w",
        )
        for var in ["x1", "x2"]:
            row = result[result["Variable"] == var].iloc[0]
            npt.assert_allclose(row["Difference"], S[f"diff_{var}"], atol=1e-3)
            npt.assert_allclose(row["SMD"], S[f"smd_{var}"], atol=1e-3)
            npt.assert_allclose(row["Variance Ratio"], S[f"vr_{var}"], atol=1e-3)

    def test_unweighted_path_unchanged(self, df_balance_weighted):
        """Calling balance() without weights should still work."""
        from open_econs.models.causal.balance import balance

        result = balance(
            df_balance_weighted,
            treatment="t",
            covariates=["x1", "x2"],
            weights=None,
        )
        expected_cols = [
            "Variable",
            "Treated Mean",
            "Control Mean",
            "Difference",
            "Treated Std",
            "Control Std",
            "t-statistic",
            "P>|t|",
        ]
        assert result.columns.tolist() == expected_cols
        assert "SMD" not in result.columns
        assert "Variance Ratio" not in result.columns
