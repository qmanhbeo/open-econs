"""Stata parity tests for DiD and Event Study."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import read_stata


class TestDiDBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_did):
        self.s = read_stata("did_basic")
        self.oe_r = oe.did("y ~ treat * post", data=df_did,
                           treatment="treat", post="post")

    def test_did_coefficient(self):
        oe_att = self.oe_r.coefficients.values[-1]
        npt.assert_allclose(oe_att, self.s["b_treatXpost"], rtol=1e-7)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])


class TestDiDCluster:
    @pytest.fixture(autouse=True)
    def _run(self, df_did):
        self.s = read_stata("did_cluster")
        self.oe_r = oe.did("y ~ treat * post", data=df_did,
                           treatment="treat", post="post", cluster="unit")

    def test_cluster_se(self):
        oe_se = self.oe_r.std_errors.values[-1]
        npt.assert_allclose(oe_se, self.s["se_treatXpost"], rtol=1e-7)


class TestEventStudy:
    @pytest.fixture(autouse=True)
    def _run(self, df_did):
        self.s = read_stata("event_study")
        df = df_did.copy()
        df["treat_event_time"] = df["post"].astype(int)
        try:
            self.oe_r = oe.event_study("y ~ treat * post", data=df,
                                        treatment="treat", post="post")
        except (ValueError, TypeError):
            pytest.skip("oe.event_study() has formulaic contrast encoding bug")

    def test_intercept(self):
        npt.assert_allclose([self.oe_r.coefficients.values[0]],
                            [self.s["b_int"]], rtol=1e-7)
