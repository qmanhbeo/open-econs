"""R parity tests for ARDL/UECM + PSS(2001) bounds test.

Fixture
-------
``tests/r/fixtures/expected/ardl.json`` stores the ground truth from R's
``ARDL`` package (v0.2.5) for the canonical ``denmark`` example:

    LRM ~ LRY + IBO + IDE, order = c(3, 1, 3, 2), PSS case 3.

It contains:

- ``f_stat`` / ``t_stat``  -- bounds F / t statistics
- ``ec_term``              -- coef on the lagged level of LRM (L(LRM, 1))
- ``lr_LRY`` / ``lr_IBO`` / ``lr_IDE`` -- long-run multipliers
- ``f_cv_lower_*`` / ``f_cv_upper_*`` -- PSS(2001) Table CI F I(0)/I(1)
- ``t_cv_lower_*`` / ``t_cv_upper_*`` -- PSS(2001) Table CII t I(0)/I(1)

Per the parity standard, p-values are NEVER asserted cross-tool.  Only the
statistics, the EC term, the long-run multipliers, and the PSS(2001)
critical values are compared to 1e-6.
"""

from __future__ import annotations

import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..r_runner import read_r

pytestmark = pytest.mark.r

# Module-level fixture cache
R = read_r("ardl")
DF = pd.read_csv("tests/r/fixtures/inputs/ardl_input.csv")

ATOL = 1e-6


class TestARDLUECMBoundsR:
    """OE uecm_fit + bounds_test vs R ARDL package, PSS case 3, k=3."""

    @pytest.fixture(scope="class")
    def uecm_result(self):
        return oe.uecm_fit(
            DF,
            "LRM",
            exog=["LRY", "IBO", "IDE"],
            order={"LRY": 1, "IBO": 3, "IDE": 2},
            lags=3,
            trend="c",
        )

    @pytest.fixture(scope="class")
    def bounds(self, uecm_result):
        return uecm_result.bounds_test(3, cv_vintage="pss2001")

    def test_f_stat(self, bounds):
        npt.assert_allclose(bounds.f_stat, R["f_stat"], rtol=0, atol=ATOL)

    def test_t_stat(self, bounds):
        npt.assert_allclose(bounds.t_stat, R["t_stat"], rtol=0, atol=ATOL)

    def test_ec_term(self, uecm_result):
        npt.assert_allclose(uecm_result.ec_term, R["ec_term"], rtol=0, atol=ATOL)

    def test_long_run_LRY(self, uecm_result):
        npt.assert_allclose(uecm_result.long_run["LRY"], R["lr_LRY"], rtol=0, atol=ATOL)

    def test_long_run_IBO(self, uecm_result):
        npt.assert_allclose(uecm_result.long_run["IBO"], R["lr_IBO"], rtol=0, atol=ATOL)

    def test_long_run_IDE(self, uecm_result):
        npt.assert_allclose(uecm_result.long_run["IDE"], R["lr_IDE"], rtol=0, atol=ATOL)

    def test_f_crit_lower(self, bounds):
        npt.assert_allclose(bounds.f_crit_lower["10%"], R["f_cv_lower_10"], rtol=0, atol=ATOL)
        npt.assert_allclose(bounds.f_crit_lower["5%"], R["f_cv_lower_5"], rtol=0, atol=ATOL)
        npt.assert_allclose(bounds.f_crit_lower["1%"], R["f_cv_lower_1"], rtol=0, atol=ATOL)

    def test_f_crit_upper(self, bounds):
        npt.assert_allclose(bounds.f_crit_upper["10%"], R["f_cv_upper_10"], rtol=0, atol=ATOL)
        npt.assert_allclose(bounds.f_crit_upper["5%"], R["f_cv_upper_5"], rtol=0, atol=ATOL)
        npt.assert_allclose(bounds.f_crit_upper["1%"], R["f_cv_upper_1"], rtol=0, atol=ATOL)

    def test_t_crit_lower(self, bounds):
        npt.assert_allclose(bounds.t_crit_lower["10%"], R["t_cv_lower_10"], rtol=0, atol=ATOL)
        npt.assert_allclose(bounds.t_crit_lower["5%"], R["t_cv_lower_5"], rtol=0, atol=ATOL)
        npt.assert_allclose(bounds.t_crit_lower["1%"], R["t_cv_lower_1"], rtol=0, atol=ATOL)

    def test_t_crit_upper(self, bounds):
        npt.assert_allclose(bounds.t_crit_upper["10%"], R["t_cv_upper_10"], rtol=0, atol=ATOL)
        npt.assert_allclose(bounds.t_crit_upper["5%"], R["t_cv_upper_5"], rtol=0, atol=ATOL)
        npt.assert_allclose(bounds.t_crit_upper["1%"], R["t_cv_upper_1"], rtol=0, atol=ATOL)
