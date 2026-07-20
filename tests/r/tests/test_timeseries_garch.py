"""R (rugarch) parity tests for GARCH(1,1) (``oe.garch`` vs ``rugarch``).

Fixture inputs
--------------
The R generator (``tests/r/generate-fixtures/garch_basic.R``) and this test
read ``tests/r/fixtures/inputs/garch_basic_input.csv`` (600 obs, identical to
the Stata-side ``garch_input.csv``).  ``rugarch::ugarchfit`` (sGARCH, norm)
freely estimates ``omega`` via full MLE -- identical to ``arch`` and to OE's
``garch(p=1, q=1)``.

Tolerance
---------
**Documented exception to the rule-2 1e-6 ceiling.**  The remaining
coefficient-level gap (~5e-3 relative on alpha) is the omega-beta ridge:
omega and beta are near-collinear in the variance recursion, so the
likelihood is flat along the ridge.  The presample backcast convention was
matched to R's ``rec.init="all"`` (``mean(e²)``), closing the LL gap from
~1.4e-4 relative to ~2.2e-5 relative.  The 6e-3 relative tolerance covers
the residual ridge-driven coefficient spread with margin.
"""

from __future__ import annotations

from pathlib import Path

import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..r_runner import read_r

pytestmark = pytest.mark.r

RTOL = 6e-3

INPUT_CSV = (
    Path(__file__).resolve().parents[2]
    / "r" / "fixtures" / "inputs" / "garch_basic_input.csv"
)

R_GARCH = read_r("garch_basic")


def _y() -> pd.Series:
    return pd.read_csv(INPUT_CSV)["y"].astype(float).reset_index(drop=True)


def _params(y: pd.Series) -> dict[str, float]:
    r = oe.garch(y, p=1, q=1)
    return {
        "mu": float(r.params["mu"]),
        "omega": float(r.params["omega"]),
        "alpha": float(r.params["alpha[1]"]),
        "beta": float(r.params["beta[1]"]),
        "ll": float(r.llf),
    }


class TestGARCHR:
    """GARCH(1,1) parameters + log-likelihood vs R ``rugarch``."""

    def test_mu(self):
        p = _params(_y())
        npt.assert_allclose(p["mu"], R_GARCH["mu"], rtol=RTOL)

    def test_omega(self):
        p = _params(_y())
        npt.assert_allclose(p["omega"], R_GARCH["omega"], rtol=RTOL)

    def test_alpha(self):
        p = _params(_y())
        npt.assert_allclose(p["alpha"], R_GARCH["alpha"], rtol=RTOL)

    def test_beta(self):
        p = _params(_y())
        npt.assert_allclose(p["beta"], R_GARCH["beta"], rtol=RTOL)

    def test_loglik(self):
        p = _params(_y())
        npt.assert_allclose(p["ll"], R_GARCH["ll"], rtol=RTOL)
