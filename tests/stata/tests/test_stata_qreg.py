"""Stata parity tests for quantile_reg() against committed .dta fixtures.

Ground truth: ``tests/stata/fixtures/expected/qreg.dta`` (``qreg`` coefs + default
SEs at tau in {0.25, 0.5}) and ``qreg_boot.dta`` (``bsqreg``/``sqreg`` bootstrap
SEs).  Coefficients and the default analytic SE reproduce Stata ``qreg`` to
<=1e-6 (rule 2).  Bootstrap SEs are asserted only to a documented tolerance
because Stata's bootstrap RNG is not portable (the *point estimates* remain
exact; see methodology/linear/quantile.md).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe
from ..stata_runner import read_stata

INPUT = pd.read_csv("tests/stata/fixtures/inputs/df_qreg.csv")

STATA_TOL = 1e-6
BOOT_TOL = 1e-2  # bootstrap RNG not portable; documented tolerance only


class TestStataCoefficients:
    def test_qreg_beta_q50_matches_stata(self):
        s = read_stata("qreg")
        r = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.5, se_method="stata")
        npt.assert_allclose(r.coefficients["Intercept"], s["b0_q50"], atol=STATA_TOL)
        npt.assert_allclose(r.coefficients["x1"], s["b1_q50"], atol=STATA_TOL)
        npt.assert_allclose(r.coefficients["x2"], s["b2_q50"], atol=STATA_TOL)

    def test_qreg_beta_q25_matches_stata(self):
        s = read_stata("qreg")
        r = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.25, se_method="stata")
        npt.assert_allclose(r.coefficients["Intercept"], s["b0_q25"], atol=STATA_TOL)
        npt.assert_allclose(r.coefficients["x1"], s["b1_q25"], atol=STATA_TOL)
        npt.assert_allclose(r.coefficients["x2"], s["b2_q25"], atol=STATA_TOL)

    def test_nobs_matches(self):
        s = read_stata("qreg")
        r = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.5)
        assert r.nobs == int(s["N_q50"])


class TestStataDefaultSE:
    """Stata ``qreg`` DEFAULT VCE == ``se_method='stata'`` (rule-15 default)."""

    def test_qreg_se_q50_matches_stata(self):
        s = read_stata("qreg")
        r = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.5, se_method="stata")
        npt.assert_allclose(r.std_errors["Intercept"], s["se0_q50"], atol=STATA_TOL)
        npt.assert_allclose(r.std_errors["x1"], s["se1_q50"], atol=STATA_TOL)
        npt.assert_allclose(r.std_errors["x2"], s["se2_q50"], atol=STATA_TOL)

    def test_qreg_se_q25_matches_stata(self):
        s = read_stata("qreg")
        r = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.25, se_method="stata")
        npt.assert_allclose(r.std_errors["Intercept"], s["se0_q25"], atol=STATA_TOL)
        npt.assert_allclose(r.std_errors["x1"], s["se1_q25"], atol=STATA_TOL)
        npt.assert_allclose(r.std_errors["x2"], s["se2_q25"], atol=STATA_TOL)

    def test_vcov_diagonal_matches_V(self):
        s = read_stata("qreg")
        r = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.5, se_method="stata")
        diag = np.diag(r.vcov().values)
        # vcov rows/cols follow [Intercept, x1, x2] == [se0, se1, se2].
        npt.assert_allclose(
            np.sqrt(diag),
            [s["se0_q50"], s["se1_q50"], s["se2_q50"]],
            atol=STATA_TOL,
        )


class TestStataBootstrapSE:
    """Bootstrap SEs: coefficients exact (deterministic BR simplex), SEs checked
    for plausibility only. Stata's bootstrap RNG is not portable, so the Python
    side (NumPy default_rng) cannot reproduce Stata's exact bootstrap SEs; we
    assert they are positive and within an order of magnitude of the analytic
    Stata SE (documented tolerance). See methodology/linear/quantile.md.
    """

    def test_bsqreg_coefs_match_stata(self):
        s = read_stata("qreg_boot")
        r = oe.quantile_reg(
            "y ~ x1 + x2", INPUT, method="bsqreg", seed=20260719, reps=20
        )
        npt.assert_allclose(r.coefficients["Intercept"], s["bsq_b0_q50"], atol=STATA_TOL)
        npt.assert_allclose(r.coefficients["x1"], s["bsq_b1_q50"], atol=STATA_TOL)
        npt.assert_allclose(r.coefficients["x2"], s["bsq_b2_q50"], atol=STATA_TOL)

    def test_bsqreg_se_plausible(self):
        s = read_stata("qreg_boot")
        r = oe.quantile_reg(
            "y ~ x1 + x2", INPUT, method="bsqreg", seed=20260719, reps=20
        )
        # Positive and within an order of magnitude of Stata's bootstrap SE.
        # Stata fixture keys: se0=Intercept, se1=x1, se2=x2.
        for col, idx in (("Intercept", 0), ("x1", 1), ("x2", 2)):
            se_py = r.std_errors[col]
            se_st = s[f"bsq_se{idx}_q50"]
            assert se_py > 0
            assert abs(se_py - se_st) <= BOOT_TOL + 10 * se_st

    def test_sqreg_coefs_match_stata(self):
        s = read_stata("qreg_boot")
        # sqreg with a single tau is numerically bsqreg; request tau=0.5.
        r = oe.quantile_reg(
            "y ~ x1 + x2", INPUT, method="sqreg", seed=20260719, reps=20, tau=0.5
        )
        npt.assert_allclose(r.coefficients["Intercept"], s["sq_b0_q50"], atol=STATA_TOL)
        npt.assert_allclose(r.coefficients["x1"], s["sq_b1_q50"], atol=STATA_TOL)
        npt.assert_allclose(r.coefficients["x2"], s["sq_b2_q50"], atol=STATA_TOL)

    def test_sqreg_se_plausible(self):
        s = read_stata("qreg_boot")
        r = oe.quantile_reg(
            "y ~ x1 + x2", INPUT, method="sqreg", seed=20260719, reps=20, tau=0.5
        )
        for col, idx in (("Intercept", 0), ("x1", 1), ("x2", 2)):
            se_py = r.std_errors[col]
            se_st = s[f"sq_se{idx}_q50"]
            assert se_py > 0
            assert abs(se_py - se_st) <= BOOT_TOL + 10 * se_st
