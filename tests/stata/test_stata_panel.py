"""Stata parity tests for Panel estimators."""

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
    run_do(label)
    df = pd.read_stata(DO_DIR / f"{label}.dta")
    return dict(zip(df["name"], df["value"]))


class TestPanelFE:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = _stata("panel_fe")
        self.oe_r = oe.fe("y ~ x + z", data=df_panel, entity="entity", time="time")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]], rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]], rtol=1e-6)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])

    def test_r_squared(self):
        assert abs(self.oe_r.r_squared - self.s["r2_w"]) < 1e-4


class TestPanelRE:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = _stata("panel_re")
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        self.oe_r = ctx.re("y ~ x + z")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]], rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]], rtol=1e-6)


class TestPanelPooled:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = _stata("panel_pooled")
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        self.oe_r = ctx.pooled("y ~ x + z")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]], rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]], rtol=1e-6)


class TestPanelFD:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = _stata("panel_fd")
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        self.oe_r = ctx.diff("y ~ x + z")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]], rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]], rtol=1e-6)


class TestPanelHausman:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = _stata("panel_hausman")
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        fe_r = ctx.fe("y ~ x + z")
        re_r = ctx.re("y ~ x + z")
        self.oe_h = ctx.hausman(fe_r, re_r)

    def test_chi2(self):
        assert abs(self.oe_h.statistic - self.s["chi2"]) < 0.5

    def test_p_value(self):
        assert abs(self.oe_h.p_value - self.s["p"]) < 0.1
