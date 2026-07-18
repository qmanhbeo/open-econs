"""Stata parity tests for the ARDL/UECM PSS bounds test.

Source-verified against ``c:\\ado\\plus\\a\\ardl.ado`` (v1.0.6) and the
PSS (2001) published critical-value table (``ardlbounds, table nosurfreg``,
``nsource=pssmith`` — NOT Narayan, since no ``n()`` option is passed).

Canonical example: denmark data ``LRM ~ LRY + IBO + IDE`` with
``lags(3 1 3 2)``, PSS case 3 (unrestricted constant).

e(b) layout (EC representation)
-------------------------------
``ardl ..., ec`` returns ``e(b)`` with three equation blocks:

  * ``ADJ:``  — the EC / speed-of-adjustment term, coef on ``L.depvar``
               (here ``L.lrm``); position 1 of ``e(b)``.
  * ``LR:``   — long-run (level) coefficients on the level regressors
               ``L.lry L.ibo L.ide``; positions 2,3,4 of ``e(b)``.
  * ``SR:``   — short-run (difference) coefficients.

The LR coefficients are equation-qualified (``LR:L.lry`` etc.) and cannot be
addressed with bare ``_b[L.lry]``; the fixture extracts them by matrix
position.

Critical-value matrix layout
----------------------------
``e(F_critval)`` and ``e(t_critval)`` are 1 x 8:

  col 1=10%lo 2=10%hi 3=5%lo 4=5%hi 5=2.5%lo 6=2.5%hi 7=1%lo 8=1%hi

For case 3 (k=3) the PSS(2001) F- and t-tables contain a 2.5% row, and Stata
returns it (stored as ``f_cv_lower_25`` / ``f_cv_upper_25`` / ``t_cv_lower_25``
/ ``t_cv_upper_25`` in the fixture).  OE exposes 2.5% when ``bounds_test`` is
called with ``signif`` including ``0.025``; the 2.5% class below does exactly
that and asserts it at 1e-6.

Full-precision fixture (rule 18 footgun — resolved)
---------------------------------------------------
Every asserted quantity — ``e(F_pss)``, ``e(t_pss)``, the EC term, the LR
multipliers, and ALL critical values — matches OE (= R = statsmodels) to well
within 1e-6.  An earlier apparent ~1e-5 Stata-vs-statsmodels gap on F/t/EC/LR
was a *fixture-generation* artifact: ``import delimited`` defaults to reading
numeric columns as single-precision ``float``, truncating the (near-collinear,
R^2=0.988) inputs.  The generator now issues ``set type double`` before import,
restoring machine-precision parity.  See ``methodology/timeseries/ardl.md``.
All assertions are at ``atol=1e-6`` — nothing is loosened.
"""

from __future__ import annotations

import numpy.testing as npt
import pandas as pd
import pytest

from open_econs.models.timeseries.ardl import uecm_fit

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

# Module-level fixture cache (Stata as ground truth)
STATA = read_stata("ardl")
DF = pd.read_csv("tests/r/fixtures/inputs/ardl_input.csv")


def _fit(signif=(0.10, 0.05, 0.01)):
    """Estimate the UECM and run the PSS bounds test (case 3)."""
    model = uecm_fit(
        DF,
        "LRM",
        exog=["LRY", "IBO", "IDE"],
        order={"LRY": 1, "IBO": 3, "IDE": 2},
        lags=3,
        trend="c",
    )
    bt = model.bounds_test(3, signif=signif)
    return model, bt


class TestStataARDLBoundsStats:
    """OE PSS F/t stats vs Stata ``ardl ..., ec`` anchors."""

    @pytest.fixture(scope="class")
    def fit(self):
        return _fit()

    def test_f_stat(self, fit):
        _, bt = fit
        npt.assert_allclose(bt.f_stat, STATA["f_stat"], rtol=0, atol=1e-6)

    def test_t_stat(self, fit):
        _, bt = fit
        npt.assert_allclose(bt.t_stat, STATA["t_stat"], rtol=0, atol=1e-6)


class TestStataARDLCoefficients:
    """OE EC term and long-run multipliers vs Stata ``ardl ..., ec``."""

    @pytest.fixture(scope="class")
    def fit(self):
        return _fit()

    def test_ec_term(self, fit):
        model, _ = fit
        npt.assert_allclose(model.ec_term, STATA["ec_term"], rtol=0, atol=1e-6)

    def test_lr_LRY(self, fit):
        model, _ = fit
        npt.assert_allclose(model.long_run["LRY"], STATA["lr_LRY"], rtol=0, atol=1e-6)

    def test_lr_IBO(self, fit):
        model, _ = fit
        npt.assert_allclose(model.long_run["IBO"], STATA["lr_IBO"], rtol=0, atol=1e-6)

    def test_lr_IDE(self, fit):
        model, _ = fit
        npt.assert_allclose(model.long_run["IDE"], STATA["lr_IDE"], rtol=0, atol=1e-6)


class TestStataARDLFCritVals:
    """OE F critical values vs Stata ``e(F_critval)`` (PSS 2001)."""

    @pytest.fixture(scope="class")
    def fit(self):
        return _fit()

    def test_f_10(self, fit):
        _, bt = fit
        npt.assert_allclose(bt.f_crit_lower["10%"], STATA["f_cv_lower_10"], rtol=0, atol=1e-6)
        npt.assert_allclose(bt.f_crit_upper["10%"], STATA["f_cv_upper_10"], rtol=0, atol=1e-6)

    def test_f_5(self, fit):
        _, bt = fit
        npt.assert_allclose(bt.f_crit_lower["5%"], STATA["f_cv_lower_5"], rtol=0, atol=1e-6)
        npt.assert_allclose(bt.f_crit_upper["5%"], STATA["f_cv_upper_5"], rtol=0, atol=1e-6)

    def test_f_1(self, fit):
        _, bt = fit
        npt.assert_allclose(bt.f_crit_lower["1%"], STATA["f_cv_lower_1"], rtol=0, atol=1e-6)
        npt.assert_allclose(bt.f_crit_upper["1%"], STATA["f_cv_upper_1"], rtol=0, atol=1e-6)


class TestStataARDLTCritVals:
    """OE t critical values vs Stata ``e(t_critval)`` (PSS 2001)."""

    @pytest.fixture(scope="class")
    def fit(self):
        return _fit()

    def test_t_10(self, fit):
        _, bt = fit
        npt.assert_allclose(bt.t_crit_lower["10%"], STATA["t_cv_lower_10"], rtol=0, atol=1e-6)
        npt.assert_allclose(bt.t_crit_upper["10%"], STATA["t_cv_upper_10"], rtol=0, atol=1e-6)

    def test_t_5(self, fit):
        _, bt = fit
        npt.assert_allclose(bt.t_crit_lower["5%"], STATA["t_cv_lower_5"], rtol=0, atol=1e-6)
        npt.assert_allclose(bt.t_crit_upper["5%"], STATA["t_cv_upper_5"], rtol=0, atol=1e-6)

    def test_t_1(self, fit):
        _, bt = fit
        npt.assert_allclose(bt.t_crit_lower["1%"], STATA["t_cv_lower_1"], rtol=0, atol=1e-6)
        npt.assert_allclose(bt.t_crit_upper["1%"], STATA["t_cv_upper_1"], rtol=0, atol=1e-6)


class TestStataARDLCritVals25:
    """OE 2.5% F- and t-bounds (``signif`` incl. 0.025) vs Stata PSS 2001."""

    @pytest.fixture(scope="class")
    def fit(self):
        return _fit(signif=(0.025,))

    def test_f_25(self, fit):
        _, bt = fit
        npt.assert_allclose(bt.f_crit_lower["2.5%"], STATA["f_cv_lower_25"], rtol=0, atol=1e-6)
        npt.assert_allclose(bt.f_crit_upper["2.5%"], STATA["f_cv_upper_25"], rtol=0, atol=1e-6)

    def test_t_25(self, fit):
        _, bt = fit
        npt.assert_allclose(bt.t_crit_lower["2.5%"], STATA["t_cv_lower_25"], rtol=0, atol=1e-6)
        npt.assert_allclose(bt.t_crit_upper["2.5%"], STATA["t_cv_upper_25"], rtol=0, atol=1e-6)
