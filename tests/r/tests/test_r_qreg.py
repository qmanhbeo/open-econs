"""R parity tests for quantile_reg() against committed .json fixtures.

Ground truth: ``tests/r/fixtures/expected/qreg.json`` produced by R's
``quantreg::rq(method="br")`` + ``summary.rq(se="ker", hs=TRUE)``.  Coefficients
reproduce R to machine precision; ``se_method='ker'`` reproduces R's Powell
kernel sandwich VCE to <=1e-6 (rule 2).  ``se_method='stata'`` intentionally
follows Stata's sparsity VCE and is covered by the Stata fixture instead.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe
from ..r_runner import read_r

INPUT = pd.read_csv(
    "C:/Users/manhn/Desktop/open-econs/tests/r/fixtures/inputs/qreg_input.csv"
)

R_TOL = 1e-6


class TestRCoefficients:
    def test_qreg_beta_q50_matches_r(self):
        r = read_r("qreg")
        res = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.5, se_method="ker")
        npt.assert_allclose(res.coefficients.values, r["q50"]["coef"], atol=R_TOL)

    def test_qreg_beta_q25_matches_r(self):
        r = read_r("qreg")
        res = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.25, se_method="ker")
        npt.assert_allclose(res.coefficients.values, r["q25"]["coef"], atol=R_TOL)


class TestRKernelSE:
    """R ``summary.rq(se="ker", hs=TRUE)`` == ``se_method='ker'``."""

    def test_qreg_se_q50_matches_r(self):
        r = read_r("qreg")
        res = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.5, se_method="ker")
        npt.assert_allclose(res.std_errors.values, r["q50"]["se"], atol=R_TOL)

    def test_qreg_se_q25_matches_r(self):
        r = read_r("qreg")
        res = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.25, se_method="ker")
        npt.assert_allclose(res.std_errors.values, r["q25"]["se"], atol=R_TOL)

    def test_vcov_diagonal_matches_r(self):
        r = read_r("qreg")
        res = oe.quantile_reg("y ~ x1 + x2", INPUT, tau=0.5, se_method="ker")
        diag = np.sqrt(np.diag(res.vcov().values))
        npt.assert_allclose(diag, r["q50"]["se"], atol=R_TOL)
