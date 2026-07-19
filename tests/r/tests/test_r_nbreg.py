"""R parity tests for ``oe.nbreg`` (NB2, pooled + fixed effects).

Source-verified against R ``fixest::fenegbin`` (FE NB2) and R ``MASS::glm.nb``
(pooled NB2).  ``oe.nbreg(dispersion="const")`` reproduces the textbook NB2
gamma-Poisson mixture to ``1e-6`` on coefficients, ``theta = 1/alpha``, and
log-likelihood.  See ``methodology/limited/nbreg.md``.

Fixtures:
    tests/r/fixtures/expected/nbreg.json  (tests/r/generate-fixtures/nbreg.R)
    tests/r/fixtures/inputs/nbreg_input.csv

All assertions at ``atol=1e-6`` — nothing loosened (rule 2).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from open_econs.models.limited.nbreg import nbreg

from ..r_runner import read_r

pytestmark = pytest.mark.r

R = read_r("nbreg")
DF = pd.read_csv("tests/r/fixtures/inputs/nbreg_input.csv")


class TestRNBRegPooledCoefficients:
    """Pooled NB2 (glm.nb) point estimates match R to 1e-6."""

    def test_b_x1(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const")
        npt.assert_allclose(r.coefficients["x1"], R["b_x1_p"], rtol=0, atol=1e-6)

    def test_b_x2(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const")
        npt.assert_allclose(r.coefficients["x2"], R["b_x2_p"], rtol=0, atol=1e-6)


class TestRNBRegPooledOverdispersion:
    """Pooled NB2 theta = 1/alpha matches R glm.nb to 1e-6."""

    def test_theta(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const")
        npt.assert_allclose(r.theta(), R["theta_p"], rtol=0, atol=1e-6)
        npt.assert_allclose(1.0 / r.alpha(), R["theta_p"], rtol=0, atol=1e-6)

    def test_loglik(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const")
        npt.assert_allclose(r.llf, R["loglik_p"], rtol=0, atol=1e-6)


class TestRNBRegFECoefficients:
    """FE NB2 (fenegbin) point estimates match R to 1e-6."""

    def test_b_x1(self):
        r = nbreg("y ~ x1 + x2", data=DF, fixed_effects=["firm", "year"])
        npt.assert_allclose(r.coefficients["x1"], R["b_x1"], rtol=0, atol=1e-6)

    def test_b_x2(self):
        r = nbreg("y ~ x1 + x2", data=DF, fixed_effects=["firm", "year"])
        npt.assert_allclose(r.coefficients["x2"], R["b_x2"], rtol=0, atol=1e-6)


class TestRNBRegFEOverdispersion:
    """FE NB2 theta matches R fenegbin to 1e-6."""

    def test_theta(self):
        r = nbreg("y ~ x1 + x2", data=DF, fixed_effects=["firm", "year"])
        npt.assert_allclose(r.theta(), R["theta"], rtol=0, atol=1e-6)
        npt.assert_allclose(r.llf, R["loglik"], rtol=0, atol=1e-6)


class TestRNBRegStdErrors:
    """Pooled NB2 robust (iid) SEs match R glm.nb OIM SEs to 1e-6."""

    def test_se_x1(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const", cov_type="nonrobust")
        npt.assert_allclose(r.std_errors["x1"], R["se_x1_p"], rtol=0, atol=1e-6)

    def test_se_x2(self):
        r = nbreg("y ~ x1 + x2", data=DF, dispersion="const", cov_type="nonrobust")
        npt.assert_allclose(r.std_errors["x2"], R["se_x2_p"], rtol=0, atol=1e-6)
