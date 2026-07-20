"""Stata parity tests for ARIMA/ARMA(1,1) (``oe.arima`` vs Stata ``arima``).

Fixture inputs
--------------
The Stata ``.do`` (``tests/stata/generate-fixtures/arima_arma11.do``) and this
test read ``tests/r/fixtures/inputs/arma_input.csv`` (300 obs).  Stata
``arima y, ar(1) ma(1)`` is pure ML (state-space Kalman), matching OE's
default ``method="ml"``.

Sign-convention check (resolved, no correction applied)
------------------------------------------------------
The AR/MA sign convention was empirically verified against Stata ``arima y,
ar(1) ma(1)`` and R ``stats::arima(y, order=c(1,0,1))``: all three agree
exactly on coefficient signs and magnitudes (AR=0.6875, MA=-0.5679,
const=-0.009, LL=-419.316).  The historical statsmodels/R sign flip applied
only to the *deprecated* ``tsa.arima_model.ARMA`` MLE path; the current
statespace implementation already matches Stata and R, so no sign correction
is applied in the wrapper (documented per standing rule 1).  These tests
therefore assert AR/MA/const/LL directly with no sign remap.
"""

from __future__ import annotations

from pathlib import Path

import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

# LL agrees across OE / Stata / R to ~2e-9, so 1e-6 is the principled bound.
RTOL_LL = 1e-6

# Documented exception (rule 2 ceiling intentionally exceeded -- see
# docs/timeseries-backend-recon.md, "ARIMA flat-likelihood exception"). The
# ARMA(1,1) likelihood is flat in the AR/MA subspace near the optimum: shifting
# the coefficients by the observed ~4e-5 changes the log-likelihood by only
# ~2e-9. Independent optimizers (statsmodels / Stata / R) therefore land at
# numerically distinct points in this flat basin (Stata vs R even differ by ~1e-5
# on ar1). The coefficient is not identified to 1e-6 by the likelihood; 1e-4
# bounds the genuine cross-tool spread with margin.
RTOL_COEF = 1e-4

# The constant is a near-zero parameter (~0.009); relative tolerance is
# inappropriate and it too is subject to the same flat-likelihood spread
# (genuine cross-tool absolute gap ~1.8e-5). 1e-4 absolute bounds it with margin.
ATOL_CONST = 1e-4

REPO_ROOT = Path(__file__).resolve().parents[3]
ARMA_INPUT = REPO_ROOT / "tests" / "r" / "fixtures" / "inputs" / "arma_input.csv"

S_ARIMA = read_stata("arima_arma11")


def _y() -> pd.Series:
    return pd.read_csv(ARMA_INPUT)["y"].astype(float).reset_index(drop=True)


def _fit() -> dict[str, float]:
    r = oe.arima(_y(), order=(1, 0, 1), trend="c")
    return {
        "cons": float(r.params["const"]),
        "ar1": float(r.params["ar.L1"]),
        "ma1": float(r.params["ma.L1"]),
        "ll": float(r.llf),
    }


class TestARIMAStata:
    """ARMA(1,1) coefficients + log-likelihood vs Stata ``arima``."""

    def test_const(self):
        f = _fit()
        npt.assert_allclose(f["cons"], S_ARIMA["cons"], rtol=0, atol=ATOL_CONST)

    def test_ar1(self):
        f = _fit()
        npt.assert_allclose(f["ar1"], S_ARIMA["ar1"], rtol=RTOL_COEF)

    def test_ma1(self):
        f = _fit()
        npt.assert_allclose(f["ma1"], S_ARIMA["ma1"], rtol=RTOL_COEF)

    def test_loglik(self):
        f = _fit()
        npt.assert_allclose(f["ll"], S_ARIMA["ll"], rtol=RTOL_LL)
