"""Stata parity tests for Panel estimators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import read_stata


class TestPanelFE:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = read_stata("panel_fe")
        # Stata `xtreg y x z, fe` is one-way entity FE only
        self.oe_r = oe.fe("y ~ x + z", data=df_panel, entity="entity",
                          cov_type="nonrobust")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]],
                            rtol=1e-6)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])

    def test_r_squared(self):
        npt.assert_allclose(self.oe_r.r_squared, self.s["r2_w"], rtol=1e-6)


class TestPanelFETwoWay:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = read_stata("panel_fe_twoway")
        # Stata `xtreg y x z i.time, fe` — entity FE + time dummies
        self.oe_r = oe.fe("y ~ x + z", data=df_panel, entity="entity",
                          time="time", cov_type="nonrobust")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]],
                            rtol=1e-6)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])

    def test_df_resid(self):
        assert self.oe_r.df_resid == int(self.s["df_r"])

    def test_r_squared(self):
        npt.assert_allclose(self.oe_r.r_squared, self.s["r2_w"], rtol=1e-6)


class TestPanelRE:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = read_stata("panel_re")
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        # linearmodels uses "unadjusted" not "nonrobust"
        self.oe_r = ctx.re("y ~ x + z", cov_type="unadjusted")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_int"], self.s["b_x"], self.s["b_z"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_int"], self.s["se_x"], self.s["se_z"]],
                            rtol=1e-6)


class TestPanelPooled:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = read_stata("panel_pooled")
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        self.oe_r = ctx.pooled("y ~ x + z", cov_type="nonrobust")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_int"], self.s["b_x"], self.s["b_z"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_int"], self.s["se_x"], self.s["se_z"]],
                            rtol=1e-6)


class TestPanelFD:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = read_stata("panel_fd")
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        self.oe_r = ctx.diff("y ~ x + z")

    def test_coefficients(self):
        # oe uses linearmodels FirstDifferenceOLS vs Stata manual diff+regress
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]],
                            rtol=5e-5)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]],
                            rtol=1e-2)


class TestPanelHausman:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = read_stata("panel_hausman")
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        # One-way entity FE to match Stata's xtreg y x z, fe
        fe_r = ctx.fe("y ~ x + z", cov_type="nonrobust", entity="entity", time=None)
        # linearmodels uses "unadjusted" not "nonrobust"
        re_r = ctx.re("y ~ x + z", cov_type="unadjusted")
        self.oe_h = ctx.hausman(fe_r, re_r)

    def test_chi2(self):
        npt.assert_allclose(self.oe_h.statistic, self.s["chi2"], rtol=1e-6)

    def test_p_value(self):
        npt.assert_allclose(self.oe_h.p_value, self.s["p"], rtol=1e-6)


class TestFEVcovIndexConsistency:
    """Regression: fe().coefficients.index must equal fe().vcov().index.

    Prevents recurrence of the bug where sm.OLS on a bare numpy array
    auto-generated ['x1','x2'] names that desynced vcov() from
    coefficients, breaking ctx.hausman().
    """

    @pytest.fixture(autouse=True)
    def _data(self):
        self.df = pd.read_csv(
            str(Path(__file__).parent / "fixtures" / "df_panel.csv")
        )

    def test_entity_only(self):
        r = oe.fe("y ~ x + z", data=self.df, entity="entity", cov_type="nonrobust")
        pd.testing.assert_index_equal(r.coefficients.index, r.vcov().index)

    def test_time_only(self):
        r = oe.fe("y ~ x + z", data=self.df, time="time", cov_type="HC2")
        pd.testing.assert_index_equal(r.coefficients.index, r.vcov().index)

    def test_two_way(self):
        r = oe.fe(
            "y ~ x + z", data=self.df, entity="entity", time="time",
            cov_type="nonrobust",
        )
        pd.testing.assert_index_equal(r.coefficients.index, r.vcov().index)

    def test_clustered(self):
        r = oe.fe(
            "y ~ x + z", data=self.df, entity="entity",
            cluster="entity", cov_type="HC2",
        )
        pd.testing.assert_index_equal(r.coefficients.index, r.vcov().index)
