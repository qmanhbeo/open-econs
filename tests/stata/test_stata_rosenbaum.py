"""Stata parity test for Rosenbaum bounds sensitivity analysis.

Validates ``rosenbaum_bounds()`` against Stata's ``rbounds`` (Gangl, v1.1.6)
on a small controlled pair-difference dataset that deliberately includes two
zero-difference pairs (exercising the zero-handling convention).

The fixture follows Stata's zero-difference convention:
  - Zero-diff pairs are included in the rank computation (they occupy the
    lowest rank positions).
  - Zero-diff pairs contribute 0 to T, E[T], and V[T] via psp=0, psm=0.

Reference
---------
Stata: ``rbounds`` (Markus Gangl, v1.1.6, SSC).
Python: ``rosenbaum_bounds()`` in ``open_econs.models.causal.sensitivity``.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from open_econs.models.causal.sensitivity import rosenbaum_bounds
from .stata_runner import read_stata

S = read_stata("rosenbaum_pairs")

# Pair differences from the Stata fixture
_DIFFS = [1.5, 2.0, -0.5, 0.0, 3.0, -1.0, 0.5, -2.0, 0.0, 1.0]


class TestRosenbaumPairs:
    """One-sided p-values vs Stata rbounds at Γ = 1, 2, 3."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.s = S
        self.result = rosenbaum_bounds(_DIFFS, gamma_max=3.0, gamma_inc=1.0)

    def test_n_pairs(self):
        assert self.result._n_pairs == int(self.s["N"])

    def test_gamma1_sig_plus(self):
        row = self.result.tidy().query("abs(Gamma - 1.0) < 1e-12")
        npt.assert_allclose(
            row["upper_bound_p"].values[0],
            self.s["g1_sigp"],
            rtol=1e-6,
        )

    def test_gamma1_sig_minus(self):
        row = self.result.tidy().query("abs(Gamma - 1.0) < 1e-12")
        npt.assert_allclose(
            row["lower_bound_p"].values[0],
            self.s["g1_sigm"],
            rtol=1e-6,
        )

    def test_gamma2_sig_plus(self):
        row = self.result.tidy().query("abs(Gamma - 2.0) < 1e-12")
        npt.assert_allclose(
            row["upper_bound_p"].values[0],
            self.s["g2_sigp"],
            rtol=1e-6,
        )

    def test_gamma2_sig_minus(self):
        row = self.result.tidy().query("abs(Gamma - 2.0) < 1e-12")
        npt.assert_allclose(
            row["lower_bound_p"].values[0],
            self.s["g2_sigm"],
            rtol=1e-6,
        )

    def test_gamma3_sig_plus(self):
        row = self.result.tidy().query("abs(Gamma - 3.0) < 1e-12")
        npt.assert_allclose(
            row["upper_bound_p"].values[0],
            self.s["g3_sigp"],
            rtol=1e-6,
        )

    def test_gamma3_sig_minus(self):
        row = self.result.tidy().query("abs(Gamma - 3.0) < 1e-12")
        npt.assert_allclose(
            row["lower_bound_p"].values[0],
            self.s["g3_sigm"],
            rtol=1e-6,
        )

    def test_critical_gamma_type(self):
        cg = self.result.critical_gamma
        assert cg is None or isinstance(cg, float)
