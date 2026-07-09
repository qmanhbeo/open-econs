"""Stata parity tests for OLS estimators."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import run_do, stata_available, DO_DIR

pytestmark = pytest.mark.skipif(
    not stata_available(), reason="StataMP not found"
)


def _stata(label: str) -> dict[str, float]:
    """Run a .do file and read its .dta output as a dict."""
    run_do(label)
    df = pd.read_stata(DO_DIR / f"{label}.dta")
    return dict(zip(df["name"], df["value"]))


class TestOLSBasic:
    def test_coefficients(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        s = _stata("ols_basic")
        npt.assert_allclose(oe_r.coefficients.values,
                            [s["b_int"], s["b_x1"], s["b_x2"]], rtol=1e-6)

    def test_standard_errors(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        s = _stata("ols_basic")
        npt.assert_allclose(oe_r.std_errors.values,
                            [s["se_int"], s["se_x1"], s["se_x2"]], rtol=1e-6)

    def test_nobs(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        s = _stata("ols_basic")
        assert oe_r.nobs == int(s["N"])

    def test_df_resid(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        s = _stata("ols_basic")
        assert oe_r.df_resid == int(s["df_r"])

    def test_r_squared(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        s = _stata("ols_basic")
        assert abs(oe_r.r_squared - s["r2"]) < 1e-6

    def test_adj_r_squared(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        s = _stata("ols_basic")
        assert abs(oe_r.adj_r_squared - s["r2_a"]) < 1e-6

    def test_f_statistic(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        s = _stata("ols_basic")
        assert abs(oe_r.f_statistic - s["F"]) < 1e-4


class TestOLSRobust:
    def test_se(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="HC1")
        s = _stata("ols_robust")
        npt.assert_allclose(oe_r.std_errors.values,
                            [s["se_int"], s["se_x1"], s["se_x2"]], rtol=1e-6)


class TestOLSCluster:
    def test_se(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cluster="province")
        s = _stata("ols_cluster")
        npt.assert_allclose(oe_r.std_errors.values,
                            [s["se_int"], s["se_x1"], s["se_x2"]], rtol=1e-6)


class TestOLSHAC:
    def test_se(self, df_ols):
        df = df_ols.copy()
        df["time"] = range(len(df))
        oe_r = oe.ols("y ~ x1 + x2", data=df, cov_type="HAC", lags=2, time="time")
        s = _stata("ols_hac")
        # HAC implementations differ slightly between packages (~1%)
        npt.assert_allclose(oe_r.std_errors.values,
                            [s["se_int"], s["se_x1"], s["se_x2"]], rtol=1e-2)


class TestOLSPredict:
    def test_predict_first_10(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        oe_pred = oe_r.predict().values[:10]
        run_do("ols_predict")
        stata_pred = pd.read_stata(DO_DIR / "ols_predict.dta")
        npt.assert_allclose(oe_pred, stata_pred["yhat"].values, rtol=1e-6)


class TestOLSConfInt:
    def test_conf_int(self, df_ols):
        oe_r = oe.ols("y ~ x1 + x2", data=df_ols, cov_type="nonrobust")
        s = _stata("ols_confint")
        npt.assert_allclose(oe_r.conf_int["lower"].values,
                            [s["b_int_ll"], s["b_x1_ll"], s["b_x2_ll"]], rtol=1e-6)
        npt.assert_allclose(oe_r.conf_int["upper"].values,
                            [s["b_int_ul"], s["b_x1_ul"], s["b_x2_ul"]], rtol=1e-6)
