"""Stata parity tests for ADF and Phillips-Perron unit-root tests.

Fixture inputs
--------------
Both the Stata ``.do`` generators (``tests/stata/generate-fixtures/ur_adf_*.do``,
``ur_pp_*.do``) and these tests read the *same* canonical series
``tests/r/fixtures/inputs/ur_input.csv`` (all ``ur_*.csv`` inputs are byte-
identical 200-observation copies, verified).  Stata ``dfuller`` / ``pperron``
emit ``r(Zt)`` (the test statistic) and the MacKinnon (1994) approximate
p-value ``r(p)`` / ``r(pval)``.

Why the p-value, not the CV table, is the anchor
------------------------------------------------
arch (OE's backend) prints MacKinnon (2010) critical values; Stata prints
Fuller (1976); R prints banded Fuller.  The *critical-value tables* therefore
diverge in small samples and are NOT a valid cross-tool equality target
(standing rule 2: never paper over a convention mismatch with a loose
tolerance).  The one quantity that genuinely agrees across ``arch`` and Stata
is the **MacKinnon (1994) approximate p-value**, which Stata surfaces as
"MacKinnon approximate p-value" and arch computes natively.  We therefore
assert the statistic AND the p-value against Stata, and regression-test only
the CV-vintage *label* (same-backend, not cross-tool) elsewhere.

The R side carries the statistic-only anchor (urca provides no MacKinnon
p-value) -- see ``tests/r/tests/test_timeseries_urca.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..stata_runner import read_stata

pytestmark = pytest.mark.stata

# Tightened to 1e-6 per the v1.1.0 tolerance-standard re-audit: OE matches
# Stata's ADF/PP statistic AND MacKinnon p-value to <=8e-9 (empirically
# confirmed), so the maximally-tight 1e-6 is the principled bound (rule 2).
RTOL = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[3]
UR_INPUT = REPO_ROOT / "tests" / "r" / "fixtures" / "inputs" / "ur_input.csv"

S_ADF_C = read_stata("ur_adf_c")
S_ADF_CT = read_stata("ur_adf_ct")
S_PP_C = read_stata("ur_pp_c")
S_PP_CT = read_stata("ur_pp_ct")


def _y() -> pd.Series:
    return pd.read_csv(UR_INPUT)["y"].astype(float).reset_index(drop=True)


class TestADFStata:
    """ADF vs Stata ``dfuller`` -- statistic + MacKinnon p-value."""

    def test_c_statistic(self):
        oe_r = oe.adf(_y(), lags=0, trend="c")
        npt.assert_allclose(oe_r.stat, S_ADF_C["stat"], rtol=RTOL)

    def test_c_pvalue(self):
        oe_r = oe.adf(_y(), lags=0, trend="c")
        npt.assert_allclose(oe_r.pvalue, S_ADF_C["pvalue"], rtol=RTOL)

    def test_ct_statistic(self):
        oe_r = oe.adf(_y(), lags=0, trend="ct")
        npt.assert_allclose(oe_r.stat, S_ADF_CT["stat"], rtol=RTOL)

    def test_ct_pvalue(self):
        oe_r = oe.adf(_y(), lags=0, trend="ct")
        npt.assert_allclose(oe_r.pvalue, S_ADF_CT["pvalue"], rtol=RTOL)

    def test_cv_vintage_label(self):
        # Regression guard: same-backend only -- arch's MacKinnon (2010) table.
        oe_r = oe.adf(_y(), lags=0, trend="c")
        assert "MacKinnon" in oe_r.cv_vintage

    @pytest.mark.xfail(
        strict=True,
        reason="TS-1 (FUTURE_WORK line ~252): OE ADF CV = arch MacKinnon (2010);"
        " Stata dfuller CV = interpolated Fuller (1976). 5% quantile differs by"
        " ~7.2e-3 >> 1e-6 until the Fuller-vintage table is ported. The "
        "statistic and MacKinnon p-value (asserted above) remain the genuine "
        "cross-tool anchors.",
    )
    def test_cv5_matches_stata_fuller(self):
        # TS-1 gap: the tabulated 5% CV diverges from Stata's Fuller table even
        # though the statistic/p-value agree to <=8e-9 (standing rule 2: the
        # divergence is captured, not papered over with a loose tolerance).
        oe_r = oe.adf(_y(), lags=0, trend="c")
        npt.assert_allclose(
            oe_r.critical_values["5%"], S_ADF_C["cv_5"], atol=1e-6
        )


class TestPPStata:
    """PP vs Stata ``pperron`` -- statistic + MacKinnon p-value.

    OE's ``pp`` default ``bandwidth="stata"`` = ``int(4*(T/100)**(2/9))``, which
    matches Stata ``pperron``'s default Newey-West bandwidth exactly.
    """

    def test_c_statistic(self):
        oe_r = oe.pp(_y(), trend="c")
        npt.assert_allclose(oe_r.stat, S_PP_C["stat"], rtol=RTOL)

    def test_c_pvalue(self):
        oe_r = oe.pp(_y(), trend="c")
        npt.assert_allclose(oe_r.pvalue, S_PP_C["pvalue"], rtol=RTOL)

    def test_ct_statistic(self):
        oe_r = oe.pp(_y(), trend="ct")
        npt.assert_allclose(oe_r.stat, S_PP_CT["stat"], rtol=RTOL)

    def test_ct_pvalue(self):
        oe_r = oe.pp(_y(), trend="ct")
        npt.assert_allclose(oe_r.pvalue, S_PP_CT["pvalue"], rtol=RTOL)

    def test_cv_vintage_label(self):
        oe_r = oe.pp(_y(), trend="c")
        assert "MacKinnon" in oe_r.cv_vintage
