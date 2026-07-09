"""Stata parity tests for DiD and Event Study."""

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


class TestDiDBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_did):
        self.s = _stata("did_basic")
        self.oe_r = oe.did("y ~ treat * post", data=df_did,
                           treatment="treat", post="post")

    def test_did_coefficient(self):
        oe_att = self.oe_r.coefficients.values[-1]
        assert abs(oe_att - self.s["b_treatXpost"]) < 1e-6

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])


class TestDiDCluster:
    @pytest.fixture(autouse=True)
    def _run(self, df_did):
        self.s = _stata("did_cluster")
        self.oe_r = oe.did("y ~ treat * post", data=df_did,
                           treatment="treat", post="post", cluster="unit")

    def test_cluster_se(self):
        oe_se = self.oe_r.std_errors.values[-1]
        assert abs(oe_se - self.s["se_treatXpost"]) < 1e-4


class TestEventStudy:
    @pytest.fixture(autouse=True)
    def _run(self, df_did):
        self.s = _stata("event_study")
        self.oe_r = oe.event_study("y ~ treat * post", data=df_did,
                                   treatment="treat", post="post")

    def test_intercept(self):
        npt.assert_allclose([self.oe_r.coefficients.values[0]],
                            [self.s["b_int"]], rtol=1e-6)
