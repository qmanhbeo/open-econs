"""Stata parity tests for Oaxaca-Blinder (SSC: oaxaca)."""

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


class TestOaxacaTwoFold:
    @pytest.fixture(autouse=True)
    def _run(self, df_oaxaca):
        self.s = _stata("oaxaca_two_fold")
        self.oe_r = oe.oaxaca("y ~ edu + age + female", data=df_oaxaca,
                               by="female", decomposition_type="two-fold")

    def test_total_gap(self):
        assert abs(self.oe_r.total_gap - self.s["gap"]) < 1e-4

    def test_explained(self):
        assert abs(self.oe_r.explained - self.s["explained"]) < 1e-4


class TestOaxacaThreeFold:
    @pytest.fixture(autouse=True)
    def _run(self, df_oaxaca):
        self.s = _stata("oaxaca_three_fold")
        self.oe_r = oe.oaxaca("y ~ edu + age + female", data=df_oaxaca,
                               by="female", decomposition_type="three-fold")

    def test_total_gap(self):
        assert abs(self.oe_r.total_gap - self.s["gap"]) < 1e-4
