"""Stata parity tests for the McCrary / Cattaneo-Jansson-Ma density test.

Both the ``backend="rddensity"`` wrapper and the from-source ``backend="builtin"``
path are validated *directly* against Stata's ``rddensity`` output (not merely
against each other), satisfying the project's independent-parity requirement.
The bandwidths selected by Stata are fed to both paths so the comparison
isolates the estimator math - the same convention used by
``tests/stata/test_stata_rdd.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

import open_econs as oe
from open_econs.models.causal.rdd import _RDENSITY

from ..stata_runner import INPUTS_DIR, read_stata

pytestmark = pytest.mark.stata

import sys as _sys

_THIS = _sys.modules[__name__]


@pytest.fixture(scope="session", autouse=True)
def _load_stata_density_fixture() -> None:
    """Load the Stata ``rddensity`` reference at setup, not import time.

    Running Stata via ``read_stata`` during collection disrupted sibling
    conftest fixture registration when this module was collected before
    others in ``tests/stata/``.
    """
    s = read_stata("rdd_density")
    _THIS.S = s
    _THIS.H = (s["h_l"], s["h_r"])
    yield


def _load() -> pd.DataFrame:
    return pd.read_csv(INPUTS_DIR / "df_rdd_density.csv")


@pytest.mark.skipif(not _RDENSITY, reason="rddensity package not installed")
class TestDensityStataBackend:
    @pytest.fixture(autouse=True)
    def _run(self):
        self.df = _load()
        self.oe = oe.density_test(self.df, "x", 0.0, backend="rddensity", h=H)

    def test_theta(self):
        assert abs(self.oe.theta - S["theta"]) < 1e-6

    def test_se(self):
        assert abs(self.oe.se - S["se"]) < 1e-6

    def test_z(self):
        assert abs(self.oe.z_stat - S["z"]) < 1e-6

    def test_p(self):
        assert abs(self.oe.p_value - S["p"]) < 1e-6

    def test_bandwidth(self):
        assert abs(self.oe.h_left - S["h_l"]) < 1e-8
        assert abs(self.oe.h_right - S["h_r"]) < 1e-8

    def test_one_sided_densities(self):
        assert abs(self.oe.fhat_left - S["fhat_l"]) < 1e-6
        assert abs(self.oe.fhat_right - S["fhat_r"]) < 1e-6

    def test_n(self):
        assert self.oe.n_left == int(S["n_l"])
        assert self.oe.n_right == int(S["n_r"])


class TestDensityStataBuiltin:
    """From-source estimator, validated directly against Stata rddensity."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.df = _load()
        self.oe = oe.density_test(self.df, "x", 0.0, backend="builtin", h=H)

    def test_theta(self):
        assert abs(self.oe.theta - S["theta"]) < 1e-6

    def test_se(self):
        assert abs(self.oe.se - S["se"]) < 1e-6

    def test_z(self):
        assert abs(self.oe.z_stat - S["z"]) < 1e-6

    def test_p(self):
        assert abs(self.oe.p_value - S["p"]) < 1e-6

    def test_bandwidth(self):
        assert abs(self.oe.h_left - S["h_l"]) < 1e-8
        assert abs(self.oe.h_right - S["h_r"]) < 1e-8

    def test_one_sided_densities(self):
        assert abs(self.oe.fhat_left - S["fhat_l"]) < 1e-6
        assert abs(self.oe.fhat_right - S["fhat_r"]) < 1e-6

    def test_n(self):
        assert self.oe.n_left == int(S["n_l"])
        assert self.oe.n_right == int(S["n_r"])
