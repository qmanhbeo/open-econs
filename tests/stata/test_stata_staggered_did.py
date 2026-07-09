"""Stata parity tests for Staggered DiD (SSC: csdid)."""

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


class TestStaggeredDiD:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = _stata("staggered_did")
        df = df_panel.copy()
        # Create binary treatment indicator (on from treatment period onward)
        # Match the Stata .do: entities 10-19 treat at t=3, entities 20+ treat at t=5
        df["treat"] = 0.0
        df.loc[(df["entity"] >= 10) & (df["entity"] < 20) & (df["time"] >= 3), "treat"] = 1.0
        df.loc[(df["entity"] >= 20) & (df["time"] >= 5), "treat"] = 1.0
        self.oe_r = oe.staggered_did(df, y="y", entity="entity",
                                     time="time", treatment="treat")

    def test_att(self):
        # OLS-based approx vs doubly-robust csdid → relaxed tolerance
        assert abs(self.oe_r.att - self.s["ATT"]) < 1.0
