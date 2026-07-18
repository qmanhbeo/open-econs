"""ARDL / UECM + PSS bounds test -- internal backend-identity regression test.

Placement rationale (rule 7)
----------------------------
This file lives in ``non_stata_nor_r/`` because the asserted quantities are
**internal OE-vs-statsmodels computational identities** plus guards on the
OE-embedded published PSS(2001) critical-value tables.  Cross-tool parity vs
Stata ``ardl`` / R ``ARDL`` is asserted separately in
``tests/stata/tests/test_stata_ardl.py`` and ``tests/r/tests/test_r_ardl.py``.

What is pinned here
-------------------
* OE ``ardl_fit`` / ``uecm_fit`` reproduce ``statsmodels.tsa.ardl`` params,
  SEs, IC and residuals byte-for-byte (the wrapper adds nothing to the fit).
* The F-**statistic** from ``bounds_test`` matches statsmodels for all 5 PSS
  cases (statsmodels' F-stat is convention-free; only its CV table is
  simulation-based, which is why OE serves the published table instead).
* ``lr_sign`` toggle: ``"stata"`` == ``-"statsmodels"`` (sign flip) with the
  ``y.L1`` normalization base dropped.
* ``cv_vintage`` toggle: ``"pss2001"`` returns the published PSS table
  (verified against a spot value); ``"statsmodels"`` returns the simulated
  bounds; the F-statistic is identical across both settings.
* The embedded ``_PSS_F_BOUNDS`` / ``_PSS_T_BOUNDS`` tables are internally
  consistent (I(0) <= I(1) for F, I(0) >= I(1) for the negative t-bounds, and
  the k=0 row has equal lower/upper by construction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from statsmodels.tsa.ardl import ARDL, UECM

import open_econs as oe
from open_econs.models.timeseries.ardl import _PSS_F_BOUNDS, _PSS_T_BOUNDS

REPO_ROOT = Path(__file__).resolve().parents[2]
ARDL_INPUT = REPO_ROOT / "tests" / "r" / "fixtures" / "inputs" / "ardl_input.csv"

# Canonical Pesaran denmark example: ARDL(3,1,3,2), case 3.
Y = "LRM"
EXOG = ["LRY", "IBO", "IDE"]
ORDER = {"LRY": 1, "IBO": 3, "IDE": 2}
LAGS = 3


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(ARDL_INPUT)


@pytest.fixture(scope="module")
def endog_exog(df):
    return df[Y].astype(float), df[EXOG].astype(float)


class TestARDLBackendIdentity:
    """OE ardl_fit must reproduce statsmodels ARDL exactly."""

    def test_params_identical(self, df, endog_exog):
        endog, exog = endog_exog
        oe_r = oe.ardl_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        sm = ARDL(endog, lags=LAGS, exog=exog, order=ORDER, trend="c").fit()
        npt.assert_allclose(oe_r.params.values, sm.params.values, rtol=0, atol=1e-12)

    def test_bse_identical(self, df, endog_exog):
        endog, exog = endog_exog
        oe_r = oe.ardl_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        sm = ARDL(endog, lags=LAGS, exog=exog, order=ORDER, trend="c").fit()
        npt.assert_allclose(oe_r.std_errors.values, sm.bse.values, rtol=0, atol=1e-12)

    def test_ic_identical(self, df, endog_exog):
        endog, exog = endog_exog
        oe_r = oe.ardl_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        sm = ARDL(endog, lags=LAGS, exog=exog, order=ORDER, trend="c").fit()
        npt.assert_allclose(oe_r.aic, sm.aic, rtol=0, atol=1e-12)
        npt.assert_allclose(oe_r.bic, sm.bic, rtol=0, atol=1e-12)


class TestUECMBackendIdentity:
    """OE uecm_fit must reproduce statsmodels UECM exactly."""

    def test_params_identical(self, df, endog_exog):
        endog, exog = endog_exog
        oe_r = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        sm = UECM(endog, lags=LAGS, exog=exog, order=ORDER, trend="c").fit()
        npt.assert_allclose(oe_r.params.values, sm.params.values, rtol=0, atol=1e-12)

    def test_ec_term_is_level_y_lag(self, df, endog_exog):
        endog, exog = endog_exog
        oe_r = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        sm = UECM(endog, lags=LAGS, exog=exog, order=ORDER, trend="c").fit()
        assert oe_r.ec_term_name == "LRM.L1"
        npt.assert_allclose(oe_r.ec_term, sm.params["LRM.L1"], rtol=0, atol=1e-12)


class TestLongRunSignToggle:
    """lr_sign: 'stata' == -'statsmodels' on the shared (non-base) terms."""

    def test_sign_flip(self, df):
        u_stata = oe.uecm_fit(
            df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c", lr_sign="stata"
        )
        u_sm = oe.uecm_fit(
            df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c", lr_sign="statsmodels"
        )
        common = u_stata.long_run.index.intersection(u_sm.long_run.index)
        assert len(common) > 0
        npt.assert_allclose(
            u_stata.long_run[common].values,
            -u_sm.long_run[common].values,
            rtol=0, atol=1e-12,
        )

    def test_stata_sign_matches_multipliers(self, df):
        # Published R multipliers() values (denmark, order 3,1,3,2).
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        npt.assert_allclose(u.long_run["LRY"], 0.9964676, rtol=0, atol=1e-6)
        npt.assert_allclose(u.long_run["IBO"], -4.5381160, rtol=0, atol=1e-6)
        npt.assert_allclose(u.long_run["IDE"], 2.8915201, rtol=0, atol=1e-6)


class TestBoundsFStatIdentity:
    """The F-statistic matches statsmodels for all 5 PSS cases."""

    @pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
    def test_f_stat_matches_statsmodels(self, df, endog_exog, case):
        endog, exog = endog_exog
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        bt = u.bounds_test(case)
        sm = UECM(endog, lags=LAGS, exog=exog, order=ORDER, trend="c").fit()
        sm_bt = sm.bounds_test(case=case, asymptotic=True)
        npt.assert_allclose(bt.f_stat, float(sm_bt.stat), rtol=0, atol=1e-10)

    def test_f_stat_invariant_to_cv_vintage(self, df):
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        a = u.bounds_test(3, cv_vintage="pss2001")
        b = u.bounds_test(3, cv_vintage="statsmodels")
        assert a.f_stat == b.f_stat


class TestCvVintageToggle:
    """pss2001 serves the published table; statsmodels serves simulated CVs."""

    def test_pss2001_matches_published_table(self, df):
        # denmark case 3, k=3 published PSS Table CI: 5% -> (3.23, 4.35).
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        bt = u.bounds_test(3, cv_vintage="pss2001")
        assert bt.k == 3
        npt.assert_allclose(bt.f_crit_lower["5%"], 3.23, rtol=0, atol=1e-12)
        npt.assert_allclose(bt.f_crit_upper["5%"], 4.35, rtol=0, atol=1e-12)
        npt.assert_allclose(bt.f_crit_upper["1%"], 5.61, rtol=0, atol=1e-12)

    def test_statsmodels_differs_from_published(self, df):
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        pub = u.bounds_test(3, cv_vintage="pss2001")
        sim = u.bounds_test(3, cv_vintage="statsmodels")
        # The simulated finite-sample upper bound is materially different.
        assert abs(pub.f_crit_upper["1%"] - sim.f_crit_upper["1%"]) > 0.1

    def test_invalid_vintage_raises(self, df):
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        with pytest.raises(ValueError, match="cv_vintage"):
            u.bounds_test(3, cv_vintage="narayan2005")

    def test_invalid_case_raises(self, df):
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        with pytest.raises(ValueError, match="case"):
            u.bounds_test(6)


class TestTBoundsCaseFolding:
    """Restricted cases fold onto their unrestricted sibling for the t-bounds."""

    def test_case_2_folds_to_3(self, df):
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        assert u.bounds_test(2).t_case == 3
        assert u.bounds_test(3).t_case == 3

    def test_case_4_folds_to_5(self, df):
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        assert u.bounds_test(4).t_case == 5
        assert u.bounds_test(5).t_case == 5

    def test_t_bounds_match_published(self, df):
        # case 3, k=3 published PSS Table CII: 5% -> (-2.86, -3.78).
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        bt = u.bounds_test(3)
        npt.assert_allclose(bt.t_crit_lower["5%"], -2.86, rtol=0, atol=1e-12)
        npt.assert_allclose(bt.t_crit_upper["5%"], -3.78, rtol=0, atol=1e-12)


class TestArdlToBoundsConversion:
    """bounds_test on an ARDLResult converts to UECM and gives the same F."""

    def test_ardl_and_uecm_agree(self, df):
        a = oe.ardl_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        u = oe.uecm_fit(df, Y, exog=EXOG, order=ORDER, lags=LAGS, trend="c")
        npt.assert_allclose(
            a.bounds_test(3).f_stat, u.bounds_test(3).f_stat, rtol=0, atol=1e-10
        )


class TestPublishedTableIntegrity:
    """Guard the embedded PSS tables against edit/transcription regressions."""

    def test_f_all_cases_k_range(self):
        assert set(_PSS_F_BOUNDS) == {1, 2, 3, 4, 5}
        for case in _PSS_F_BOUNDS:
            assert set(_PSS_F_BOUNDS[case]) == set(range(11))

    def test_t_folded_cases_k_range(self):
        assert set(_PSS_T_BOUNDS) == {1, 3, 5}
        for case in _PSS_T_BOUNDS:
            assert set(_PSS_T_BOUNDS[case]) == set(range(11))

    def test_f_lower_le_upper(self):
        for case, rows in _PSS_F_BOUNDS.items():
            for k, row in rows.items():
                for lvl, (lo, hi) in row.items():
                    assert lo <= hi + 1e-12, (case, k, lvl, lo, hi)

    def test_f_k0_bounds_equal(self):
        # At k=0 the I(0) and I(1) bounds coincide by construction.
        for case, rows in _PSS_F_BOUNDS.items():
            for lvl, (lo, hi) in rows[0].items():
                assert lo == hi, (case, lvl, lo, hi)

    def test_t_bounds_negative_and_ordered(self):
        # t-bounds are negative; the I(1) upper is more negative than I(0).
        for case, rows in _PSS_T_BOUNDS.items():
            for k, row in rows.items():
                for lvl, (lo, hi) in row.items():
                    assert lo < 0 and hi < 0, (case, k, lvl)
                    assert hi <= lo + 1e-12, (case, k, lvl, lo, hi)

    def test_monotone_upper_in_significance(self):
        # More extreme significance -> more extreme F upper bound.
        for case, rows in _PSS_F_BOUNDS.items():
            for k, row in rows.items():
                assert row["10%"][1] <= row["5%"][1] + 1e-12
                assert row["5%"][1] <= row["1%"][1] + 1e-12
