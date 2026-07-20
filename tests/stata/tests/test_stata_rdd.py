"""Stata parity tests for RDD (SSC: rdrobust).

With the rdrobust backend (default), OE matches Stata to machine precision.
Tests feed the Stata-computed bandwidth to ``oe.rdd()`` so that the only
difference is the coefficient and SE estimators — which should be identical
when the same specification (separate-side local linear, triangular kernel)
and variance estimator (NN) are used.
"""

from __future__ import annotations

import pytest

import open_econs as oe

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

S_RDD_SHARP = read_stata("rdd_sharp")
S_RDD_FUZZY = read_stata("rdd_fuzzy")


class TestRDDSharp:
    @pytest.fixture(autouse=True)
    def _run(self, df_rdd):
        self.s = S_RDD_SHARP
        self.oe_r = oe.rdd(df_rdd, y="y_sharp", running="x", cutoff=0.0,
                           bandwidth=self.s["bw"])

    def test_coef(self):
        assert abs(self.oe_r.effect - self.s["coef"]) < 5e-8

    def test_se(self):
        assert abs(self.oe_r.se - self.s["se"]) < 1e-7

    def test_bandwidth(self):
        assert abs(self.oe_r.bandwidth - self.s["bw"]) < 5e-8


class TestRDDFuzzy:
    @pytest.fixture(autouse=True)
    def _run(self, df_rdd):
        self.s = S_RDD_FUZZY
        self.oe_r = oe.rdd(df_rdd, y="y_fuzzy", running="x", cutoff=0.0,
                           treatment="treat", fuzzy=True,
                           bandwidth=self.s["bw"])

    def test_coef(self):
        assert abs(self.oe_r.effect - self.s["coef"]) < 5e-8

    def test_se(self):
        assert abs(self.oe_r.se - self.s["se"]) < 1e-7
