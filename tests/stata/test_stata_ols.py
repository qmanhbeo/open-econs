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
    def test_se(self, df_ols):
        df = df_ols.copy()
        df["time"] = range(len(df))
        oe_r = oe.ols("y ~ x1 + x2", data=df, cov_type="HAC", lags=2, time="time")
        npt.assert_allclose(oe_r.std_errors.values,
                            [S_HAC["se_int"], S_HAC["se_x1"], S_HAC["se_x2"]], rtol=1e-2)


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
