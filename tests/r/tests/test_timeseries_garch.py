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
All three engines agree to within optimizer noise (~2-3 decimals).  The 2%
relative tolerance is the principled cross-tool bound (see the Stata-side
module docstring for the full rationale; standing rule 2).  The R-side anchor
is the ``rugarch`` fit; the Stata-side anchor is ``tests/stata/...``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..r_runner import read_r

pytestmark = pytest.mark.r

RTOL = 2e-2

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
