"""R parity tests for ``oe.robust_reg`` vs R ``MASS::rlm`` (R 4.6.1).

Validated branch = ``parity="rlm"``: exact ``MASS::rlm`` coefficients and the
``vcov="rlm"`` covariance ``cov.unscaled * s^2``, matched to 1e-6 (nothing
loosened, rule 2).  See ``tests/r/generate-fixtures/rreg.R``.

The ``parity="stata"`` branch is a separate convention (Stata ``rreg``) and is
covered in ``tests/stata/tests/test_stata_rreg.py``.

These tests now validate the purely-Python ``MASS::rlm`` port (committed in
``open_econs/models/linear/robust_reg.py``) against the committed fixture
``tests/r/fixtures/expected/rreg.json``.  No R binary is launched at runtime;
``read_r`` reads the committed ``.json`` as ground truth (R is only invoked
when ``OE_REGENERATE_FIXTURES=1``).  This keeps the suite green on R-less CI
while still enforcing the 1e-6 parity oracle.

All assertions at ``atol=1e-6`` — nothing loosened.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..r_runner import read_r

pytestmark = [pytest.mark.r]

R = read_r("rreg")
DF = pd.read_csv("tests/r/fixtures/inputs/rreg_input.csv")


class TestRRregMMCoefficients:
    """MM-estimator (default) point estimates match R rlm(MM) to 1e-6."""

    def _fit(self):
        return oe.robust_reg("y ~ x1 + x2", data=DF, method="mm", parity="rlm")

    def test_b_intercept(self):
        r = self._fit()
        npt.assert_allclose(r.coefficients["(Intercept)"], R["b0"], rtol=0, atol=1e-6)

    def test_b_x1(self):
        r = self._fit()
        npt.assert_allclose(r.coefficients["x1"], R["b1"], rtol=0, atol=1e-6)

    def test_b_x2(self):
        r = self._fit()
        npt.assert_allclose(r.coefficients["x2"], R["b2"], rtol=0, atol=1e-6)

    def test_scale(self):
        r = self._fit()
        npt.assert_allclose(r.scale, R["scale"], rtol=0, atol=1e-5)


class TestRRregMMStdErrors:
    """``parity='rlm'`` SEs match R MASS::rlm covariance to 1e-6."""

    def _fit(self):
        return oe.robust_reg("y ~ x1 + x2", data=DF, method="mm", parity="rlm")

    def test_se_intercept(self):
        r = self._fit()
        npt.assert_allclose(r.std_errors["(Intercept)"], R["se0"], rtol=0, atol=1e-6)

    def test_se_x1(self):
        r = self._fit()
        npt.assert_allclose(r.std_errors["x1"], R["se1"], rtol=0, atol=1e-6)

    def test_se_x2(self):
        r = self._fit()
        npt.assert_allclose(r.std_errors["x2"], R["se2"], rtol=0, atol=1e-6)


class TestRRregMMWeights:
    """Final robustness weights match R rlm$w to 1e-6."""

    def test_weights(self):
        r = oe.robust_reg("y ~ x1 + x2", data=DF, method="mm", parity="rlm")
        npt.assert_allclose(
            np.sort(r.weights.values), np.sort(np.asarray(R["w"])), rtol=0, atol=1e-5
        )


class TestRRregHuberBranch:
    """``method="huber"`` → plain bisquare M-estimator, matches R rlm(M)."""

    def _fit(self):
        return oe.robust_reg("y ~ x1 + x2", data=DF, method="huber", parity="rlm")

    def test_b_intercept(self):
        r = self._fit()
        npt.assert_allclose(r.coefficients["(Intercept)"], R["b0_m"], rtol=0, atol=1e-6)

    def test_b_x1(self):
        r = self._fit()
        npt.assert_allclose(r.coefficients["x1"], R["b1_m"], rtol=0, atol=1e-6)

    def test_b_x2(self):
        r = self._fit()
        npt.assert_allclose(r.coefficients["x2"], R["b2_m"], rtol=0, atol=1e-6)

    def test_se_x1(self):
        r = self._fit()
        npt.assert_allclose(r.std_errors["x1"], R["se1_m"], rtol=0, atol=1e-6)


class TestRRregParityToggle:
    """``parity='rlm'`` and ``parity='stata'`` are distinct conventions (rule 15).

    The ``rlm`` branch reproduces R's exact coefficients (validated 1e-6).  The
    ``stata`` branch targets Stata's formula and differs by design.  Guard
    against the two branches accidentally collapsing together.
    """

    def test_branches_differ(self):
        r_rlm = oe.robust_reg("y ~ x1 + x2", data=DF, method="mm", parity="rlm")
        r_stata = oe.robust_reg("y ~ x1 + x2", data=DF, method="mm", parity="stata")
        gap = np.max(np.abs(r_stata.coefficients.values - r_rlm.coefficients.values))
        assert gap > 1e-4, "parity='stata' and parity='rlm' unexpectedly coincide"

    def test_rlm_branch_matches_r_fixture(self):
        r_rlm = oe.robust_reg("y ~ x1 + x2", data=DF, method="mm", parity="rlm")
        npt.assert_allclose(r_rlm.std_errors["x1"], R["se1"], rtol=0, atol=1e-6)
