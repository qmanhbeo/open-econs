"""Stata parity tests for OLS estimators."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import DO_DIR, _check_drift, read_stata, run_do

# Module-level Stata caches (read once, used by all test methods)
S_BASIC = read_stata("ols_basic")
S_ROBUST = read_stata("ols_robust")
S_CLUSTER = read_stata("ols_cluster")
S_HAC = read_stata("ols_hac")
S_CONFINT = read_stata("ols_confint")


class TestOLSBasic:
    def test_coefficients(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        npt.assert_allclose(oe_r.coefficients.values,
                            [S_BASIC["b_int"], S_BASIC["b_x1"], S_BASIC["b_x2"]], rtol=1e-6)

    def test_standard_errors(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        npt.assert_allclose(oe_r.std_errors.values,
                            [S_BASIC["se_int"], S_BASIC["se_x1"], S_BASIC["se_x2"]], rtol=1e-6)

    def test_nobs(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        assert oe_r.nobs == int(S_BASIC["N"])

    def test_df_resid(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        assert oe_r.df_resid == int(S_BASIC["df_r"])

    def test_r_squared(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        npt.assert_allclose(oe_r.r_squared, S_BASIC["r2"], rtol=1e-6)

    def test_adj_r_squared(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        npt.assert_allclose(oe_r.adj_r_squared, S_BASIC["r2_a"], rtol=1e-6)

    def test_f_statistic(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        npt.assert_allclose(oe_r.f_statistic, S_BASIC["F"], rtol=1e-4)


class TestOLSRobust:
    def test_se(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="HC1")
        npt.assert_allclose(oe_r.std_errors.values,
                            [S_ROBUST["se_int"], S_ROBUST["se_x1"], S_ROBUST["se_x2"]], rtol=1e-6)


class TestOLSCluster:
    def test_se(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cluster="province")
        npt.assert_allclose(oe_r.std_errors.values,
                            [S_CLUSTER["se_int"], S_CLUSTER["se_x1"], S_CLUSTER["se_x2"]], rtol=5e-7)


class TestOLSHAC:
    """Default HAC (``hac_adjust=False``) implements the original Newey-West
    (1987) long-run variance WITHOUT Stata's small-sample ``N/(N-K)`` df
    correction.

    Stata's ``newey`` applies ``sqrt(N/(N-K))`` unconditionally, so the two SE
    sets differ by exactly that factor. We verify the relationship analytically
    (standing rule #5 / #11) rather than papering over it with a loose blanket
    tolerance. N and K are taken from the fitted model (N=200, K=3 for this
    fixture), identical to what Stata uses, so the scaled comparison is an exact
    identity up to floating-point noise.
    """

    def test_se(self, df_ols):
        df = df_ols.copy()
        df["time"] = range(len(df))
        oe_r = oe.ols("y ~ x1 + x2", data=df, cov_type="HAC", lags=2, time="time")
        se = oe_r.std_errors.values
        stata_se = [S_HAC["se_int"], S_HAC["se_x1"], S_HAC["se_x2"]]

        # Stata's df correction on the SE scale is exactly sqrt(N/(N-K)).
        # Multiplying the uncorrected SE by it must reproduce Stata's SE to
        # float precision (this is the principled replacement for rtol=1e-2).
        n_k = np.sqrt(oe_r.nobs / (oe_r.nobs - len(oe_r.coefficients)))
        npt.assert_allclose(se * n_k, stata_se, rtol=1e-6)

        # Direct uncorrected-vs-Stata comparison. By construction
        # stata_se = uncorrected_se * n_k, so the relative discrepancy is exactly
        # `1 - 1/n_k`. A tolerance of that value plus the empirically measured
        # float-noise floor of the identity above (~2e-8) is fully justified: it
        # guards against accidentally dropping or over-applying the correction.
        rel_gap = 1.0 - 1.0 / n_k
        npt.assert_allclose(se, stata_se, rtol=rel_gap + 1e-7)


class TestOLSHACAdjust:
    """hac_adjust=True should match Stata's N/(N-K)-corrected SEs at machine precision."""

    def test_se(self, df_ols):
        df = df_ols.copy()
        df["time"] = range(len(df))
        oe_r = oe.ols("y ~ x1 + x2", data=df, cov_type="HAC", lags=2, time="time",
                       hac_adjust=True)
        npt.assert_allclose(oe_r.std_errors.values,
                            [S_HAC["se_int"], S_HAC["se_x1"], S_HAC["se_x2"]], rtol=1e-7)


class TestOLSPredict:
    def test_predict_first_10(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        oe_pred = oe_r.predict().values[:10]
        run_do("ols_predict")
        _check_drift("ols_predict")
        stata_pred = pd.read_stata(DO_DIR / "ols_predict.dta")
        npt.assert_allclose(oe_pred, stata_pred["yhat"].values, rtol=1e-6)


class TestOLSConfInt:
    def test_conf_int(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        npt.assert_allclose(oe_r.conf_int["lower"].values,
                            [S_CONFINT["b_int_ll"], S_CONFINT["b_x1_ll"], S_CONFINT["b_x2_ll"]], rtol=1e-6)
        npt.assert_allclose(oe_r.conf_int["upper"].values,
                            [S_CONFINT["b_int_ul"], S_CONFINT["b_x1_ul"], S_CONFINT["b_x2_ul"]], rtol=1e-6)
