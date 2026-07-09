"""Stata parity tests for RDD (SSC: rdrobust)."""

from __future__ import annotations

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


class TestRDDSharp:
    @pytest.fixture(autouse=True)
    def _run(self, df_rdd):
        self.s = _stata("rdd_sharp")
        self.oe_r = oe.rdd(df_rdd, y="y_sharp", running="x", cutoff=0.0)

    def test_coef(self):
        assert abs(self.oe_r.effect - self.s["coef"]) < 1e-4

    def test_se(self):
        assert abs(self.oe_r.se - self.s["se"]) < 1e-4

    def test_bandwidth(self):
        assert abs(self.oe_r.bandwidth - self.s["bw"]) < 1e-4


class TestRDDFuzzy:
    @pytest.fixture(autouse=True)
    def _run(self, df_rdd):
        self.s = _stata("rdd_fuzzy")
        self.oe_r = oe.rdd(df_rdd, y="y_fuzzy", running="x", cutoff=0.0,
                           treatment="treat", fuzzy=True)

    def test_coef(self):
        assert abs(self.oe_r.effect - self.s["coef"]) < 1e-4

    def test_se(self):
        assert abs(self.oe_r.se - self.s["se"]) < 1e-4
