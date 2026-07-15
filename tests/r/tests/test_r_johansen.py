"""R (urca) parity tests for Johansen cointegration (Cases 2, 3, 4).

Structural limitation
---------------------
R ``urca::ca.jo`` cannot reach Stata Cases 1 or 5.  The ``ecdet``
argument maps as follows (source-confirmed):

- ``ecdet="none"``  → Stata Case 3 (unrestricted constant)
- ``ecdet="const"`` → Stata Case 2 (restricted constant)
- ``ecdet="trend"`` → Stata Case 4 (restricted trend)

Cases 1 (no deterministic) and 5 (unrestricted trend) have no R-side
equivalent because ``ca.jo`` requires at least a constant in the
cointegrating equation or the data itself.  These cases are marked
``null`` in the R fixture and have no test coverage here.

Fixture
-------
``tests/r/fixtures/expected/var_basic.json`` stores trace and
max-eigenvalue statistics (reversed to Stata ascending rank order)
for Cases 2, 3, 4.

R ``ca.jo`` CVs are also stored (5% column from ``@cval``).  These
are the same Osterwald-Lenum (1992) CVs that Stata uses, so they
should match OE's O-L CVs exactly.

CV source-confirmed exception
-----------------------------
R ``ca.jo``'s embedded ``@cval`` trace CV table differs from Stata's
``_vecgetcv.ado`` O-L table at certain cells.  Specifically, R's
trace layer row 1 (r<=1) is identical to its maxeigenvalue layer
row 1 (e.g., 9.24 for ``ecdet="const"``), whereas Stata's trace
CV at r=1 is 9.42 (different from maxeig's 9.24).  This is a
genuine difference in the embedded tables, not a fixture bug.
See Stata ``_vecgetcv.ado`` lines 39-80 vs R ``urca`` source
``.urcval`` / ``cv.const`` arrays.  CV parity tests are therefore
in the Stata test file only; the R test file tests statistic
parity (trace, maxeig) which does match.
"""

from __future__ import annotations

import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..r_runner import read_r

pytestmark = pytest.mark.r

# Module-level fixture cache
R_VAR = read_r("var_basic")
DF_VAR_INPUT = pd.read_csv("tests/r/fixtures/inputs/var_input.csv")

# R tolerance: R ``ca.jo`` uses the same O-L algorithm, so statistics
# should agree to ≤1e-4 (small floating-point differences from
# implementation details).
RTOL = 1e-4


class TestJohansenCase2R:
    """Case 2: R ecdet="const" → Stata trend(rconstant)."""

    @pytest.fixture(scope="class")
    def result(self):
        return oe.johansen_cointegration(
            DF_VAR_INPUT, case=2, k_ar_diff=1, signif=0.05,
        )

    def test_trace_r0(self, result):
        npt.assert_allclose(
            result.trace_stat.iloc[0], R_VAR["trace_case2"][0], rtol=RTOL,
        )

    def test_trace_r1(self, result):
        npt.assert_allclose(
            result.trace_stat.iloc[1], R_VAR["trace_case2"][1], rtol=RTOL,
        )

    def test_maxeig_r0(self, result):
        npt.assert_allclose(
            result.max_eig_stat.iloc[0], R_VAR["maxeig_case2"][0], rtol=RTOL,
        )

    def test_maxeig_r1(self, result):
        npt.assert_allclose(
            result.max_eig_stat.iloc[1], R_VAR["maxeig_case2"][1], rtol=RTOL,
        )


class TestJohansenCase3R:
    """Case 3: R ecdet="none" → Stata trend(constant)."""

    @pytest.fixture(scope="class")
    def result(self):
        return oe.johansen_cointegration(
            DF_VAR_INPUT, case=3, k_ar_diff=1, signif=0.05,
        )

    def test_trace_r0(self, result):
        npt.assert_allclose(
            result.trace_stat.iloc[0], R_VAR["trace_case3"][0], rtol=RTOL,
        )

    def test_trace_r1(self, result):
        npt.assert_allclose(
            result.trace_stat.iloc[1], R_VAR["trace_case3"][1], rtol=RTOL,
        )

    def test_maxeig_r0(self, result):
        npt.assert_allclose(
            result.max_eig_stat.iloc[0], R_VAR["maxeig_case3"][0], rtol=RTOL,
        )

    def test_maxeig_r1(self, result):
        npt.assert_allclose(
            result.max_eig_stat.iloc[1], R_VAR["maxeig_case3"][1], rtol=RTOL,
        )


class TestJohansenCase4R:
    """Case 4: R ecdet="trend" → Stata trend(rtrend)."""

    @pytest.fixture(scope="class")
    def result(self):
        return oe.johansen_cointegration(
            DF_VAR_INPUT, case=4, k_ar_diff=1, signif=0.05,
        )

    def test_trace_r0(self, result):
        npt.assert_allclose(
            result.trace_stat.iloc[0], R_VAR["trace_case4"][0], rtol=RTOL,
        )

    def test_trace_r1(self, result):
        npt.assert_allclose(
            result.trace_stat.iloc[1], R_VAR["trace_case4"][1], rtol=RTOL,
        )

    def test_maxeig_r0(self, result):
        npt.assert_allclose(
            result.max_eig_stat.iloc[0], R_VAR["maxeig_case4"][0], rtol=RTOL,
        )

    def test_maxeig_r1(self, result):
        npt.assert_allclose(
            result.max_eig_stat.iloc[1], R_VAR["maxeig_case4"][1], rtol=RTOL,
        )
