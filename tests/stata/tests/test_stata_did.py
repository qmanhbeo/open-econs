"""Stata parity tests for DiD."""

from __future__ import annotations

import numpy.testing as npt
import pytest

import open_econs as oe

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

S_DID_BASIC = read_stata("did_basic")
S_DID_CLUSTER = read_stata("did_cluster")


class TestDiDBasic:
    @pytest.fixture(autouse=True)
    def _run(self, df_did):
        self.s = S_DID_BASIC
        self.oe_r = oe.did("y ~ treat * post", data=df_did,
                           treatment="treat", post="post")

    def test_did_coefficient(self):
        oe_att = self.oe_r.coefficients.values[-1]
        npt.assert_allclose(oe_att, self.s["b_treatXpost"], rtol=1e-6)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])


class TestDiDCluster:
    @pytest.fixture(autouse=True)
    def _run(self, df_did):
        self.s = S_DID_CLUSTER
        self.oe_r = oe.did("y ~ treat * post", data=df_did,
                           treatment="treat", post="post", cluster="unit")

    def test_cluster_se(self):
        oe_se = self.oe_r.std_errors.values[-1]
        npt.assert_allclose(oe_se, self.s["se_treatXpost"], rtol=1e-6)
