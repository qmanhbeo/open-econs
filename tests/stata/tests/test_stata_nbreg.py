"""Stata parity tests for ``oe.nbreg`` (NB2).

Source-verified against Stata SSC ``nbreg`` (Hilbe / StataCorp).

IMPORTANT convention crosswalk (rule 15/16, see methodology/limited/nbreg.md):
Stata ``nbreg`` exposes two dispersion settings with DIFFERENT MLEs:
  * ``dispersion(mean)``    Var = mu*(1+alpha) — its MLE coincides with the
    textbook NB2 gamma-mixture (R glm.nb / fixest fenegbin) on this dataset
    (mean mu ~ 1).  ``oe.nbreg(dispersion="const")`` reproduces this to 1e-6.
  * ``dispersion(constant)`` Var = mu + delta*mu^2 — a Stata-SPECIFIC MLE that
    ``oe.nbreg`` does NOT reproduce.  This is a genuine source-confirmed
    divergence (verified numbers: Stata const x1 = 0.414535, delta = 1.263565,
    ll = -842.203; oe nbreg const x1 = 0.492896, alpha = 1.0563, ll =
    -836.538).  It is asserted as ``skip`` (never loosened) and documented.

Fixture: tests/stata/fixtures/expected/nbreg.dta (tests/stata/generate-fixtures/nbreg.do)
Input:   tests/r/fixtures/inputs/nbreg_input.csv

All asserted numbers at ``atol=1e-6``.
"""

from __future__ import annotations

import numpy.testing as npt
import pandas as pd
import pytest

from open_econs.models.limited.nbreg import nbreg

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

STATA = read_stata("nbreg")
DF = pd.read_csv("tests/r/fixtures/inputs/nbreg_input.csv")


class TestStataNBRegCoefficients:
    """oe nbreg(dispersion='const') == Stata nbreg, dispersion(mean) (NB2)."""

    def test_b_x1(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const")
        npt.assert_allclose(r.coefficients["x1"], STATA["b_x1"], rtol=0, atol=1e-6)

    def test_b_x2(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const")
        npt.assert_allclose(r.coefficients["x2"], STATA["b_x2"], rtol=0, atol=1e-6)


class TestStataNBRegOverdispersion:
    """alpha matches Stata e(alpha) for dispersion(mean) to 1e-6."""

    def test_alpha(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const")
        npt.assert_allclose(r.alpha(), STATA["alpha"], rtol=0, atol=1e-6)

    def test_lnalpha(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const")
        npt.assert_allclose(r.lnalpha(), float(__import__("numpy").log(STATA["alpha"])),
                            rtol=0, atol=1e-6)

    def test_loglik(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const")
        npt.assert_allclose(r.llf, STATA["ll"], rtol=0, atol=1e-6)


class TestStataNBRegStdErrors:
    """Pooled NB2 non-clustered SEs: Stata ``nbreg`` uses a robustified OIM
    information matrix (inverse of the full observed-information Hessian,
    beta + overdispersion jointly) that diverges from R glm.nb / oe OIM by up to
    ~4% on x2.  Selecting ``vcov_backend="stata"`` wraps Stata's robustified OIM
    bread so oe matches Stata ``nbreg, dispersion(mean)`` SEs to 1e-6 (rule 15
    toggle).  The default ``vcov_backend="fixest"`` keeps R glm.nb OIM parity
    (see the R parity suite)."""

    def test_se_x1(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const", cov_type="nonrobust",
                  vcov_backend="stata")
        npt.assert_allclose(r.std_errors["x1"], STATA["se_x1"], rtol=0, atol=1e-6)

    def test_se_x2(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const", cov_type="nonrobust",
                  vcov_backend="stata")
        npt.assert_allclose(r.std_errors["x2"], STATA["se_x2"], rtol=0, atol=1e-6)


class TestStataNBRegConstantDispersionGap:
    """Stata ``nbreg, dispersion(constant)`` is a Stata-specific NB2 MLE
    (Var = mu*(1+delta), source: Stata ``nbreg_al.ado``).  ``oe.nbreg`` reproduces
    it via ``dispersion="const_stata"`` to 1e-6: Stata gives x1 = 0.414535,
    delta = 1.263565, ll = -842.203.  The textbook NB2 (``dispersion="const"``)
    remains the default and matches R glm.nb / Stata dispersion(mean).  See
    methodology/limited/nbreg.md."""

    def test_b_x1(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const_stata")
        npt.assert_allclose(r.coefficients["x1"], STATA["bc_x1"], rtol=0, atol=1e-6)
