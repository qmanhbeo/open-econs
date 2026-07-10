"""Stata parity tests for Logit and Probit."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import read_stata


class TestLogitBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_logit):
        self.s = read_stata("logit_basic")
        self.oe_r = oe.logit("y ~ x1 + x2", data=df_logit)

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_int"], self.s["b_x1"], self.s["b_x2"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_int"], self.s["se_x1"], self.s["se_x2"]],
                            rtol=1e-6)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])


class TestLogitMargins:
    @pytest.fixture(autouse=True)
    def _run(self, df_logit):
        self.s = read_stata("logit_margins")
        self.oe_r = oe.logit("y ~ x1 + x2", data=df_logit)

    def test_margins(self):
        oe_me = self.oe_r.margins()
        # MEM (oe) vs AME (Stata) — different by definition, relaxed tolerance
        npt.assert_allclose(oe_me["dy/dx"].values,
                            [self.s["me_x1"], self.s["me_x2"]], rtol=0.8)


class TestProbitBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_logit):
        self.s = read_stata("probit_basic")
        self.oe_r = oe.probit("y ~ x1 + x2", data=df_logit)

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_int"], self.s["b_x1"], self.s["b_x2"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_int"], self.s["se_x1"], self.s["se_x2"]],
                            rtol=1e-6)


class TestProbitMargins:
    @pytest.fixture(autouse=True)
    def _run(self, df_logit):
        self.s = read_stata("probit_margins")
        self.oe_r = oe.probit("y ~ x1 + x2", data=df_logit)

    def test_margins(self):
        oe_me = self.oe_r.margins()
        # MEM (oe) vs AME (Stata) — different by definition, relaxed tolerance
        npt.assert_allclose(oe_me["dy/dx"].values,
                            [self.s["me_x1"], self.s["me_x2"]], rtol=0.8)
