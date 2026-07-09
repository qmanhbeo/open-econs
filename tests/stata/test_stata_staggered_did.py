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
        df["gvar"] = 0.0
        df.loc[(df["entity"] >= 10) & (df["entity"] < 20), "gvar"] = 3.0
        df.loc[df["entity"] >= 20, "gvar"] = 5.0
        self.oe_r = oe.staggered_did(df, y="y", entity="entity",
                                     time="time", treatment="gvar")

    def test_att(self):
        assert abs(self.oe_r.att - self.s["ATT"]) < 0.5
