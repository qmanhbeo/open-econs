"""Stata parity tests for Logit and Probit."""

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


class TestLogitBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_logit):
        self.s = _stata("logit_basic")
        self.oe_r = oe.logit("y ~ x1 + x2", data=df_logit)

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x1"], self.s["b_x2"]], rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x1"], self.s["se_x2"]], rtol=1e-6)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])


class TestLogitMargins:
    @pytest.fixture(autouse=True)
    def _run(self, df_logit):
        self.s = _stata("logit_margins")
        self.oe_r = oe.logit("y ~ x1 + x2", data=df_logit)

    def test_margins(self):
        oe_me = self.oe_r.margins()
        npt.assert_allclose(oe_me["dy/dx"].values,
                            [self.s["me_x1"], self.s["me_x2"]], rtol=1e-4)


class TestProbitBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_logit):
        self.s = _stata("probit_basic")
        self.oe_r = oe.probit("y ~ x1 + x2", data=df_logit)

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x1"], self.s["b_x2"]], rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x1"], self.s["se_x2"]], rtol=1e-6)


class TestProbitMargins:
    @pytest.fixture(autouse=True)
    def _run(self, df_logit):
        self.s = _stata("probit_margins")
        self.oe_r = oe.probit("y ~ x1 + x2", data=df_logit)

    def test_margins(self):
        oe_me = self.oe_r.margins()
        npt.assert_allclose(oe_me["dy/dx"].values,
                            [self.s["me_x1"], self.s["me_x2"]], rtol=1e-3)
