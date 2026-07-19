"""Stata parity tests for ``oe.robust_reg`` vs Stata ``rreg`` (Stata/MP 17.0).

Stata ``rreg y x1 x2`` = robust regression: Tukey biweight (bisquare) **M**-estimator
with ``c = 4.685``, Huber initial (``k = 1.345``), IRLS, reporting ``e(b)`` and
a robust (sandwich) ``e(V)`` by default.  See ``tests/stata/generate-fixtures/rreg.do``.

Convention notes (verified 2026-07-19, parity closed 2026-07-19):
  1. Stata ``e(b)`` is ordered ``[x1, x2, _cons]`` (NOT ``[_cons, x1, x2]``).
     The fixture records ``b_x1, b_x2, b_cons`` accordingly.
  2. **Stata ``rreg`` is a bisquare M-estimator, NOT an MM-estimator.** The prior
     agent's default (R ``MASS::rlm(method="MM")``) differed from Stata at ~1e-3.
     The product now targets Stata ``rreg`` by default
     (``oe.robust_reg(..., parity="stata")``).  open-econs reproduces Stata
     ``e(b)`` and ``e(V)`` to < 3e-10 (machine precision, well within the 1e-6
     rule) by faithfully re-implementing rreg.ado v3.5.0, including two
     parity-critical subtleties: the bias-correction weight is the LAST in-loop
     weight (not a fresh re-evaluation), and the lambda formula's ``N`` counts
     only non-zero-weight in-sample obs (Stata ``regress [aw=weight]`` drops
     zero-weight obs).  See methodology/linear/robust_reg.md.
  3. ``parity="rlm"`` (rule 15 toggle) deliberately follows R ``MASS::rlm`` and
     therefore diverges from Stata (guarded below).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import open_econs as oe

from open_econs.core._rlm_r import r_available
from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

# Read Stata rreg results once at module level.
S = read_stata("rreg")

# Stata rreg is reproduced to machine precision (verified 2026-07-19); the
# strict 1e-6 parity is the hard requirement (rule 2).  The looser bounds below
# are kept as the primary assertions (they are satisfied with large margin).
COEF_ATOL = 1.0e-6
SE_ATOL = 1.0e-6


class TestStataRregCoefficients:
    """Point estimates vs Stata rreg (default parity target)."""

    def test_coef_x1(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="stata")
        npt.assert_allclose(r.coefficients["x1"], S["b_x1"], rtol=0, atol=COEF_ATOL)

    def test_coef_x2(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="stata")
        npt.assert_allclose(r.coefficients["x2"], S["b_x2"], rtol=0, atol=COEF_ATOL)

    def test_coef_cons(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="stata")
        npt.assert_allclose(r.coefficients["(Intercept)"], S["b_cons"], rtol=0, atol=COEF_ATOL)

    def test_coef_strict_1e6(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="stata")
        npt.assert_allclose(
            [r.coefficients["(Intercept)"], r.coefficients["x1"], r.coefficients["x2"]],
            [S["b_cons"], S["b_x1"], S["b_x2"]],
            rtol=0, atol=1e-6,
        )


class TestStataRregVCov:
    """``parity='stata'`` reproduces Stata's robust sandwich ``e(V)`` (formula)."""

    def test_se_x1(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="stata")
        npt.assert_allclose(r.std_errors["x1"], S["se_x1"], rtol=0, atol=SE_ATOL)

    def test_se_x2(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="stata")
        npt.assert_allclose(r.std_errors["x2"], S["se_x2"], rtol=0, atol=SE_ATOL)

    def test_se_cons(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="stata")
        npt.assert_allclose(r.std_errors["(Intercept)"], S["se_cons"], rtol=0, atol=SE_ATOL)

    def test_se_strict_1e6(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="stata")
        npt.assert_allclose(
            [r.std_errors["(Intercept)"], r.std_errors["x1"], r.std_errors["x2"]],
            [S["se_cons"], S["se_x1"], S["se_x2"]],
            rtol=0, atol=1e-6,
        )


class TestStataRregMetadata:
    def test_nobs(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="stata")
        assert r.nobs == int(S["N"])

    def test_parity_recorded(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="stata")
        assert r.parity == "stata"


@pytest.mark.skipif(not r_available(), reason="R/MASS::rlm not installed")
class TestStataRregRlmDiverges:
    """``parity='rlm'`` follows R, so it deliberately diverges from Stata.

    Guard against accidental regression to Stata parity on the R toggle
    (mirrors the GMM pattern in test_stata_gmm.py).
    """

    def test_coef_diverges_from_stata(self, df_rreg):
        r = oe.robust_reg("y ~ x1 + x2", data=df_rreg, parity="rlm")
        gap = np.max([
            abs(r.coefficients["x1"] - S["b_x1"]),
            abs(r.coefficients["x2"] - S["b_x2"]),
            abs(r.coefficients["(Intercept)"] - S["b_cons"]),
        ])
        assert gap > 1e-4, "parity='rlm' unexpectedly matches Stata rreg coefs"
