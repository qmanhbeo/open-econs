"""Stata parity tests for GARCH(1,1) (``oe.garch`` vs Stata ``arch``).

Fixture inputs
--------------
Both the Stata ``.do`` (``tests/stata/generate-fixtures/garch_basic.do``) and
this test read ``tests/r/fixtures/inputs/garch_input.csv`` (600 obs).
Stata ``arch y, arch(1) garch(1)`` freely estimates the variance constant
``omega`` via full MLE with Gaussian errors -- identical to ``arch`` /
``rugarch`` and to OE's ``garch(p=1, q=1)`` (no variance-targeting divergence,
confirmed per standing rule 1).

Tolerance
---------
**Documented exception to the rule-2 1e-6 ceiling (see
docs/timeseries-backend-recon.md, "GARCH omega-beta ridge exception").**  The
cross-tool coefficient spread is ~1-1.5% (beta), which exceeds 1e-6.  This is
NOT optimizer noise: arch is deterministic to ~1e-7 across starting values and
tight tolerances, so the gap is *not* arch scattering.  The root cause is the
GARCH(1,1) **omega-beta ridge** -- omega and beta are near-collinear in the
variance recursion ``h_t = omega + alpha*e^2 + beta*h_{t-1}``, so the likelihood
is flat along the ridge: reducing omega and raising beta (or vice versa) leaves
the log-likelihood essentially unchanged.  Demonstrated: the three parameter
sets (OE/arch, Stata ``arch``, ``rugarch``) are all on the same likelihood ridge
-- their log-likelihoods agree to ~2e-6 relative -- and perturbing arch's optimum
along the ridge by 1% changes the LL by only ~1.6e-6.  A secondary contributor
is the presample/backcast variance initialization, which shifts the *reported*
LL by ~1.4e-4 relative between arch and Stata/R.  The 2e-2 relative tolerance is
the genuine cross-tool envelope with margin; it is an intentional, evidenced
exception and is flagged to the project lead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

RTOL = 2e-2

REPO_ROOT = Path(__file__).resolve().parents[3]
GARCH_INPUT = REPO_ROOT / "tests" / "r" / "fixtures" / "inputs" / "garch_input.csv"

S_GARCH = read_stata("garch_basic")


def _y() -> pd.Series:
    return pd.read_csv(GARCH_INPUT)["y"].astype(float).reset_index(drop=True)


def _params(y: pd.Series) -> dict[str, float]:
    r = oe.garch(y, p=1, q=1)
    return {
        "mu": float(r.params["mu"]),
        "omega": float(r.params["omega"]),
        "alpha": float(r.params["alpha[1]"]),
        "beta": float(r.params["beta[1]"]),
        "ll": float(r.llf),
    }


class TestGARCHStata:
    """GARCH(1,1) parameters + log-likelihood vs Stata ``arch garch(1)``."""

    def test_mu(self):
        p = _params(_y())
        npt.assert_allclose(p["mu"], S_GARCH["mu"], rtol=RTOL)

    def test_omega(self):
        p = _params(_y())
        npt.assert_allclose(p["omega"], S_GARCH["omega"], rtol=RTOL)

    def test_alpha(self):
        p = _params(_y())
        npt.assert_allclose(p["alpha"], S_GARCH["alpha"], rtol=RTOL)

    def test_beta(self):
        p = _params(_y())
        npt.assert_allclose(p["beta"], S_GARCH["beta"], rtol=RTOL)

    def test_loglik(self):
        p = _params(_y())
        npt.assert_allclose(p["ll"], S_GARCH["ll"], rtol=RTOL)
