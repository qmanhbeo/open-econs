"""Stata parity tests for Johansen cointegration (all 5 cases).

Fixture
-------
``tests/stata/fixtures/expected/var_basic.dta`` stores trace and
max-eigenvalue statistics for all 5 Johansen deterministic-term cases,
plus Osterwald-Lenum (1992) 5% critical values from Stata's
``_vecgetcv.ado``.

All test statistics match Stata to ≤3.6e-7 (verified via
``vecrank, lags(2) trend(...)`` for each case).  CVs match exactly.

The ``k_ar_diff`` mapping is: Stata ``vecrank, lags(2)`` → VAR order
p=2 → statsmodels ``k_ar_diff = p - 1 = 1`` (source-confirmed:
``_vecu.ado`` L89-90: ``local pm1 = 'p' - 1``).

CV-table regression test
------------------------
Case 2 O-L maxeig CV at 5% must equal 15.67 (the Stata ``rconst``
column), not 11.44 (the ``none`` column).  This protects against a
silent revert of the ``_OL_MAXEIG["rconst"]`` copy-paste bug that was
fixed in the prior session.
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

# Tolerance: ≤1e-6 for test statistics (rule 2)
RTOL = 1e-6


def _val(name: str) -> float:
    return S_VAR[name]


class TestJohansenCase1:
    """Case 1: trend(none) — no deterministic term (det_order=-1)."""

    @pytest.fixture(scope="class")
    def result(self):
        return oe.johansen_cointegration(
            DF_VAR_INPUT, case=1, k_ar_diff=1, signif=0.05,
        )

    def test_trace_r0(self, result):
        npt.assert_allclose(result.trace_stat.iloc[0], _val("trace_case1_r1"), rtol=RTOL)

    def test_trace_r1(self, result):
        npt.assert_allclose(result.trace_stat.iloc[1], _val("trace_case1_r2"), rtol=RTOL)

    def test_maxeig_r0(self, result):
        npt.assert_allclose(result.max_eig_stat.iloc[0], _val("maxeig_case1_r1"), rtol=RTOL)

    def test_maxeig_r1(self, result):
        npt.assert_allclose(result.max_eig_stat.iloc[1], _val("maxeig_case1_r2"), rtol=RTOL)

    def test_cv_trace5_r0(self, result):
        npt.assert_allclose(result.cvt.iloc[0, 1], _val("cv_trace5_case1_r1"), rtol=RTOL)

    def test_cv_trace5_r1(self, result):
        npt.assert_allclose(result.cvt.iloc[1, 1], _val("cv_trace5_case1_r2"), rtol=RTOL)

    def test_cv_maxeig5_r0(self, result):
        npt.assert_allclose(result.cvm.iloc[0, 1], _val("cv_maxeig5_case1_r1"), rtol=RTOL)

    def test_cv_maxeig5_r1(self, result):
        npt.assert_allclose(result.cvm.iloc[1, 1], _val("cv_maxeig5_case1_r2"), rtol=RTOL)


class TestJohansenCase2:
    """Case 2: trend(rconstant) — restricted constant."""

    @pytest.fixture(scope="class")
    def result(self):
        return oe.johansen_cointegration(
            DF_VAR_INPUT, case=2, k_ar_diff=1, signif=0.05,
        )

    def test_trace_r0(self, result):
        npt.assert_allclose(result.trace_stat.iloc[0], _val("trace_case2_r1"), rtol=RTOL)

    def test_trace_r1(self, result):
        npt.assert_allclose(result.trace_stat.iloc[1], _val("trace_case2_r2"), rtol=RTOL)

    def test_maxeig_r0(self, result):
        npt.assert_allclose(result.max_eig_stat.iloc[0], _val("maxeig_case2_r1"), rtol=RTOL)

    def test_maxeig_r1(self, result):
        npt.assert_allclose(result.max_eig_stat.iloc[1], _val("maxeig_case2_r2"), rtol=RTOL)

    def test_cv_trace5_r0(self, result):
        npt.assert_allclose(result.cvt.iloc[0, 1], _val("cv_trace5_case2_r1"), rtol=RTOL)

    def test_cv_trace5_r1(self, result):
        npt.assert_allclose(result.cvt.iloc[1, 1], _val("cv_trace5_case2_r2"), rtol=RTOL)

    def test_cv_maxeig5_r0(self, result):
        npt.assert_allclose(result.cvm.iloc[0, 1], _val("cv_maxeig5_case2_r1"), rtol=RTOL)

    def test_cv_maxeig5_r1(self, result):
        npt.assert_allclose(result.cvm.iloc[1, 1], _val("cv_maxeig5_case2_r2"), rtol=RTOL)


class TestJohansenCase3:
    """Case 3: trend(constant) — unrestricted constant (det_order=0)."""

    @pytest.fixture(scope="class")
    def result(self):
        return oe.johansen_cointegration(
            DF_VAR_INPUT, case=3, k_ar_diff=1, signif=0.05,
        )

    def test_trace_r0(self, result):
        npt.assert_allclose(result.trace_stat.iloc[0], _val("trace_case3_r1"), rtol=RTOL)

    def test_trace_r1(self, result):
        npt.assert_allclose(result.trace_stat.iloc[1], _val("trace_case3_r2"), rtol=RTOL)

    def test_maxeig_r0(self, result):
        npt.assert_allclose(result.max_eig_stat.iloc[0], _val("maxeig_case3_r1"), rtol=RTOL)

    def test_maxeig_r1(self, result):
        npt.assert_allclose(result.max_eig_stat.iloc[1], _val("maxeig_case3_r2"), rtol=RTOL)

    def test_cv_trace5_r0(self, result):
        npt.assert_allclose(result.cvt.iloc[0, 1], _val("cv_trace5_case3_r1"), rtol=RTOL)

    def test_cv_trace5_r1(self, result):
        npt.assert_allclose(result.cvt.iloc[1, 1], _val("cv_trace5_case3_r2"), rtol=RTOL)

    def test_cv_maxeig5_r0(self, result):
        npt.assert_allclose(result.cvm.iloc[0, 1], _val("cv_maxeig5_case3_r1"), rtol=RTOL)

    def test_cv_maxeig5_r1(self, result):
        npt.assert_allclose(result.cvm.iloc[1, 1], _val("cv_maxeig5_case3_r2"), rtol=RTOL)


class TestJohansenCase4:
    """Case 4: trend(rtrend) — restricted trend."""

    @pytest.fixture(scope="class")
    def result(self):
        return oe.johansen_cointegration(
            DF_VAR_INPUT, case=4, k_ar_diff=1, signif=0.05,
        )

    def test_trace_r0(self, result):
        npt.assert_allclose(result.trace_stat.iloc[0], _val("trace_case4_r1"), rtol=RTOL)

    def test_trace_r1(self, result):
        npt.assert_allclose(result.trace_stat.iloc[1], _val("trace_case4_r2"), rtol=RTOL)

    def test_maxeig_r0(self, result):
        npt.assert_allclose(result.max_eig_stat.iloc[0], _val("maxeig_case4_r1"), rtol=RTOL)

    def test_maxeig_r1(self, result):
        npt.assert_allclose(result.max_eig_stat.iloc[1], _val("maxeig_case4_r2"), rtol=RTOL)

    def test_cv_trace5_r0(self, result):
        npt.assert_allclose(result.cvt.iloc[0, 1], _val("cv_trace5_case4_r1"), rtol=RTOL)

    def test_cv_trace5_r1(self, result):
        npt.assert_allclose(result.cvt.iloc[1, 1], _val("cv_trace5_case4_r2"), rtol=RTOL)

    def test_cv_maxeig5_r0(self, result):
        npt.assert_allclose(result.cvm.iloc[0, 1], _val("cv_maxeig5_case4_r1"), rtol=RTOL)

    def test_cv_maxeig5_r1(self, result):
        npt.assert_allclose(result.cvm.iloc[1, 1], _val("cv_maxeig5_case4_r2"), rtol=RTOL)


class TestJohansenCase5:
    """Case 5: trend(trend) — unrestricted trend (det_order=1)."""

    @pytest.fixture(scope="class")
    def result(self):
        return oe.johansen_cointegration(
            DF_VAR_INPUT, case=5, k_ar_diff=1, signif=0.05,
        )

    def test_trace_r0(self, result):
        npt.assert_allclose(result.trace_stat.iloc[0], _val("trace_case5_r1"), rtol=RTOL)

    def test_trace_r1(self, result):
        npt.assert_allclose(result.trace_stat.iloc[1], _val("trace_case5_r2"), rtol=RTOL)

    def test_maxeig_r0(self, result):
        npt.assert_allclose(result.max_eig_stat.iloc[0], _val("maxeig_case5_r1"), rtol=RTOL)

    def test_maxeig_r1(self, result):
        npt.assert_allclose(result.max_eig_stat.iloc[1], _val("maxeig_case5_r2"), rtol=RTOL)

    def test_cv_trace5_r0(self, result):
        npt.assert_allclose(result.cvt.iloc[0, 1], _val("cv_trace5_case5_r1"), rtol=RTOL)

    def test_cv_trace5_r1(self, result):
        npt.assert_allclose(result.cvt.iloc[1, 1], _val("cv_trace5_case5_r2"), rtol=RTOL)

    def test_cv_maxeig5_r0(self, result):
        npt.assert_allclose(result.cvm.iloc[0, 1], _val("cv_maxeig5_case5_r1"), rtol=RTOL)

    def test_cv_maxeig5_r1(self, result):
        npt.assert_allclose(result.cvm.iloc[1, 1], _val("cv_maxeig5_case5_r2"), rtol=RTOL)


class TestJohansenCVRegression:
    """Regression test: Case 2 O-L maxeig CV must be the rconst column.

    Protects against a silent revert of the ``_OL_MAXEIG["rconst"]``
    copy-paste bug (was incorrectly duplicating ``"none"`` values:
    3.84/11.44 instead of 9.24/15.67).
    """

    def test_case2_maxeig_cv_is_rconst_not_none(self):
        result = oe.johansen_cointegration(
            DF_VAR_INPUT, case=2, k_ar_diff=1, signif=0.05,
        )
        # Case 2 maxeig 5% CV at K-r=2 (r=0): Stata rconst col = 15.67
        assert result.cvm.iloc[0, 1] == pytest.approx(15.67, abs=1e-6)
        # Must NOT equal the "none" column value (11.44)
        assert result.cvm.iloc[0, 1] != pytest.approx(11.44, abs=0.01)

    def test_case2_trace_cv_is_rconst(self):
        result = oe.johansen_cointegration(
            DF_VAR_INPUT, case=2, k_ar_diff=1, signif=0.05,
        )
        # Case 2 trace 5% CV at K-r=2: Stata rconst col = 19.96
        assert result.cvt.iloc[0, 1] == pytest.approx(19.96, abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Documented CV-table disparity (docs/var-vecm-backend-recon.md §2.3, "
        "Decision 1). OE's default cvt/cvm use Osterwald-Lenum (1992) to match "
        "Stata vecrank / R urca; statsmodels' native coint_johansen returns the "
        "MacKinnon-Haug-Michelis (1996) surface. Case 3, r=0, 5% trace: O-L = "
        "15.41 vs MacKinnon = 15.4943 (gap ~0.084 >> 1e-6). OE surfaces the "
        "MacKinnon table as cvt_mackinnon/cvm_mackinnon; the two are not equal. "
        "Intentionally not unified (would flip rank selection)."
    ),
)
class TestJohansenCVMackinnonDivergence:
    """OE default CVs do NOT equal statsmodels' native MacKinnon CVs.

    This is a source-confirmed convention split, not a bug. The xfail marks
    exactly what is left: a parity assertion that holds only if OE abandoned
    its O-L default. It fails today and is expected to stay failed until the
    lead decides to change the authoritative table.
    """

    def test_ol_vs_mackinnon_trace_5pc_r0(self):
        result = oe.johansen_cointegration(
            DF_VAR_INPUT, case=3, k_ar_diff=1, signif=0.05,
        )
        # If there were no disparity this would hold to 1e-6. It does not:
        npt.assert_allclose(
            result.cvt.iloc[0, 1], result.cvt_mackinnon.iloc[0, 1], rtol=1e-6,
        )

    def test_ol_vs_mackinnon_maxeig_5pc_r0(self):
        result = oe.johansen_cointegration(
            DF_VAR_INPUT, case=3, k_ar_diff=1, signif=0.05,
        )
        npt.assert_allclose(
            result.cvm.iloc[0, 1], result.cvm_mackinnon.iloc[0, 1], rtol=1e-6,
        )
