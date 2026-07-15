"""Stata parity tests for VAR IC and lag-order selection.

IC convention
-------------
OE's default ``var_fit()`` returns Stata-convention AIC/BIC/HQIC:
``AIC = ln(det(Sigma_ml)) + K*ln(2pi) + K + 2*k/T`` (matching
Stata ``estat ic`` after ``var``).  The ``lutstats`` variants
exclude deterministic terms (matching Stata ``lutstats`` option).

OE's ``var_select_order()`` uses the standard per-lag statsmodels IC
convention (no ``K*ln(2pi)+K`` offset), matching R ``VARselect`` and
statsmodels ``VAR.select_order()``.  This is for internal lag
comparison only — not directly comparable to Stata ``estat ic`` values
because the effective sample sizes differ (Lütkepohl common-sample offset
vs Stata's per-lag full-sample estimation).

Fixture
-------
``tests/stata/fixtures/expected/var_basic.dta`` stores ``ll_var``,
``aic_var``, ``bic_var``, ``hqic_var`` from ``var y1 y2, lags(1/2) /
estat ic``.  ``bic_var = e(N) * e(sbic)`` (total BIC, not per-obs).

Source
------
- Stata ``var.ado`` line 297: ``aic = -2*(ll/T) + 2*tparms/T``
- Stata ``_qsur.ado`` line 266: ``ll = -0.5*T*(K*ln(2pi) + ln(det(Sigma_ml)) + K)``
- Statsmodels ``VARResults.aic``: ``ln(det(Sigma_u_mle)) + 2*free_params/T``
- Offset ``K*ln(2pi) + K`` bridges the two (verified to 7.4e-9).
- Stata ``varsoc.ado`` lines 352-354: each lag is a separate ``var``
  estimation (same semantics as ``estat ic``, no common-sample offset).
"""

from __future__ import annotations

import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

# Module-level fixture cache
S_VAR = read_stata("var_basic")
DF_VAR_INPUT = pd.read_csv("tests/r/fixtures/inputs/var_input.csv")


class TestVARLagOrderStataIC:
    """OE ``var_fit`` IC at lag 2 vs Stata ``estat ic`` anchor points.

    Compares ``var_fit(2)`` (full-sample estimation, matching Stata's
    ``var y1 y2, lags(1/2) / estat ic`` semantics) against the Stata
    fixture.  ``var_fit`` returns Stata-convention IC values (with the
    ``K*ln(2pi)+K`` offset applied).
    """

    @pytest.fixture(scope="class")
    def result(self):
        return oe.var_fit(DF_VAR_INPUT, lags=2, trend="c")

    def test_ll_var(self, result):
        """Log-likelihood at lag 2 matches Stata."""
        npt.assert_allclose(result.llf, S_VAR["ll_var"], rtol=1e-6)

    def test_aic_var(self, result):
        """AIC at lag 2 matches Stata (all params in penalty)."""
        npt.assert_allclose(
            result.aic, S_VAR["aic_var"], rtol=1e-6,
        )

    def test_bic_var(self, result):
        """BIC at lag 2 matches Stata (total BIC = N * per-obs BIC).

        ``var_fit`` BIC is per-obs (Stata convention); fixture stores
        ``e(N) * e(sbic)`` = total BIC.
        """
        bic_total = result.bic * result.nobs
        npt.assert_allclose(bic_total, S_VAR["bic_var"], rtol=1e-6)

    def test_hqic_var(self, result):
        """HQIC at lag 2 matches Stata (all params in penalty)."""
        npt.assert_allclose(
            result.hqic, S_VAR["hqic_var"], rtol=1e-6,
        )


class TestVARLagOrderSelection:
    """OE ``var_select_order`` lag selection and IC values.

    ``var_select_order`` uses statsmodels convention (per-lag IC without
    the Stata offset) for internal lag comparison.  Selected lags are
    compared against R ``VARselect`` values (same underlying convention).
    """

    @pytest.fixture(scope="class")
    def result(self):
        return oe.var_select_order(DF_VAR_INPUT, maxlags=5, trend="c")

    def test_selected_lag_all_ics(self, result):
        """All ICs select lag 1 for this dataset (small T=200, K=2)."""
        for ic in ["aic", "bic", "hqic", "fpe"]:
            assert result.selected[ic] == 1, (
                f"Expected lag 1 for {ic}, got {result.selected[ic]}"
            )

    def test_ic_values_monotone_decrease(self, result):
        """IC values should generally decrease from lag 1 to lag 2."""
        for ic in ["aic", "bic", "hqic"]:
            vals = result.ic_values[ic]
            assert vals[1] < vals[0], (
                f"{ic}: lag 2 ({vals[1]:.6f}) should be < lag 1 ({vals[0]:.6f})"
            )
