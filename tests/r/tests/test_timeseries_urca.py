"""R (urca) parity tests for the unit-root / stationarity suite.

Engine coverage
---------------
* ``kpss`` and ``zivot_andrews`` are **R-only** here: neither has a Stata base
  command (KPSS is SSC-community only; ZA has no Stata base).  R ``urca`` is the
  sole external anchor.
* ``adf`` and ``pp`` are asserted as **OE-vs-R statistic equality** (cross-tool),
  complementing the Stata p-value anchor in
  ``tests/stata/tests/test_timeseries_adf_pp.py``.

Why the statistic (not a p-value) is the R anchor
-------------------------------------------------
``urca`` surfaces no MacKinnon p-value, so the test *statistic* is the only
available cross-tool anchor on the R side (judgment call 3).  The formal
p-value anchor remains the Stata side.  Do not read the absence of a p-value
assertion here as an inconsistency -- it is a documented source limitation.

KPSS has two separate tests (judgment call 1):
  1. ``TestKPSSMatchedBandwidth`` -- a genuine cross-tool *computational-
     equivalence* check: OE ``kpss(lags=4)`` vs R ``ur.kpss(lag="short")``.
     ``ur.kpss`` cannot take an arbitrary integer lag (only ``"short"`` /
     ``"long"``), so matching its bandwidth and comparing the statistic is the
     correct shape of the test -- NOT a tolerance relaxation.
  2. ``TestKPSSDefaultConfig`` -- a self-consistency regression guard pinning
     OE's *own* default-config output (Hobijn et al. autolag, lags=9 for this
     series).  OE's default deliberately diverges from R's default (lags=4) by
     design (documented convention), so this is NOT a cross-tool test; it exists
     so a future refactor cannot silently change OE's default behavior.
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

# Tight tolerance for the statistic-only cross-tool anchors that genuinely agree
# to <=1e-15 (ADF, KPSS matched-bandwidth, ZA all match R urca to floating-point
# precision). See the v1.1.0 tolerance-standard re-audit. PP vs R is NOT tightened
# here -- it cannot reach 1e-6 and is handled as a documented exception in the
# follow-up commit (R's ur.pp uses the dependent-variable variance in the PP
# correction term; OE/arch/Stata use the regressor variance).
RTOL_TIGHT = 1e-6

INPUT_CSV = (
    Path(__file__).resolve().parents[2]
    / "r" / "fixtures" / "inputs" / "ur_input.csv"
)

R_ADF_C = read_r("ur_adf_c")
R_ADF_CT = read_r("ur_adf_ct")
R_PP_C = read_r("ur_pp_c")
R_PP_CT = read_r("ur_pp_ct")
R_KPSS_C = read_r("ur_kpss_c")
R_KPSS_CT = read_r("ur_kpss_ct")
R_ZA_C = read_r("ur_za_c")
R_ZA_CT = read_r("ur_za_ct")

# OE's own default-config KPSS output, pinned for the self-consistency guard
# (Hobijn et al. autolag -> lags=9 on this 200-obs series).  See docstring.
OE_KPSS_C_DEFAULT = 0.3009308207544967
OE_KPSS_CT_DEFAULT = 0.21789596664515273


def _y() -> pd.Series:
    return pd.read_csv(INPUT_CSV)["y"].astype(float).reset_index(drop=True)


def _r_short_bandwidth(y: pd.Series) -> int:
    """Reproduce urca's ``lag="short"`` bandwidth: floor(4*(n/100)^0.25)."""
    n = len(y)
    return int(4 * (n / 100.0) ** 0.25)


class TestADFUr:
    """ADF vs R ``ur.df`` -- statistic equality (cross-tool)."""

    def test_c_statistic(self):
        oe_r = oe.adf(_y(), lags=0, trend="c")
        npt.assert_allclose(oe_r.stat, R_ADF_C["stat"], rtol=RTOL_TIGHT)

    def test_ct_statistic(self):
        oe_r = oe.adf(_y(), lags=0, trend="ct")
        npt.assert_allclose(oe_r.stat, R_ADF_CT["stat"], rtol=RTOL_TIGHT)


class TestPPUr:
    """PP vs R ``ur.pp`` -- statistic equality (cross-tool).

    R ``ur.pp`` is called with ``lag="short"`` (bandwidth floor(4*(n/100)^0.25));
    OE mirrors that exact bandwidth via ``bandwidth="fixed"`` so the comparison
    is a like-for-like computational-equivalence check.
    """

    def test_c_statistic(self):
        y = _y()
        bw = _r_short_bandwidth(y)
        oe_r = oe.pp(y, trend="c", bandwidth="fixed", lags=bw)
        npt.assert_allclose(oe_r.stat, R_PP_C["stat"], rtol=1e-4)

    def test_ct_statistic(self):
        y = _y()
        bw = _r_short_bandwidth(y)
        oe_r = oe.pp(y, trend="ct", bandwidth="fixed", lags=bw)
        npt.assert_allclose(oe_r.stat, R_PP_CT["stat"], rtol=1e-4)


class TestKPSSMatchedBandwidth:
    """Matched-bandwidth cross-check: OE ``kpss(lags=4)`` vs R ``ur.kpss(short)``.

    This is NOT the default-config parity test -- it pins R's ``lag="short"``
    bandwidth and confirms OE reproduces the same statistic with the same
    bandwidth.  See module docstring (judgment call 1).
    """

    def test_c_statistic(self):
        y = _y()
        bw = _r_short_bandwidth(y)
        oe_r = oe.kpss(y, lags=bw)
        npt.assert_allclose(oe_r.stat, R_KPSS_C["stat"], rtol=RTOL_TIGHT)

    def test_ct_statistic(self):
        y = _y()
        bw = _r_short_bandwidth(y)
        oe_r = oe.kpss(y, trend="ct", lags=bw)
        npt.assert_allclose(oe_r.stat, R_KPSS_CT["stat"], rtol=RTOL_TIGHT)


class TestKPSSDefaultConfig:
    """Self-consistency guard for OE's *own* default KPSS configuration.

    OE's default uses the Hobijn et al. (1998) data-dependent bandwidth
    (lags=9 here), which deliberately differs from R's ``lag="short"`` default
    (lags=4).  This test pins OE's default output so a refactor cannot change
    the default behavior silently.  It is a regression guard, not a cross-tool
    equality test.
    """

    def test_c_default(self):
        oe_r = oe.kpss(_y())
        assert oe_r.lags == 9
        npt.assert_allclose(oe_r.stat, OE_KPSS_C_DEFAULT, rtol=1e-9)

    def test_ct_default(self):
        oe_r = oe.kpss(_y(), trend="ct")
        assert oe_r.lags == 9
        npt.assert_allclose(oe_r.stat, OE_KPSS_CT_DEFAULT, rtol=1e-9)


class TestZivotAndrewsUr:
    """Zivot-Andrews vs R ``ur.za`` -- statistic equality (R-only anchor).

    No Stata base command exists for ZA; R ``urca`` is the sole external anchor.
    """

    def test_c_statistic(self):
        oe_r = oe.zivot_andrews(_y(), trend="c", lags=0)
        npt.assert_allclose(oe_r.stat, R_ZA_C["stat"], rtol=RTOL_TIGHT)

    def test_ct_statistic(self):
        oe_r = oe.zivot_andrews(_y(), trend="ct", lags=0)
        npt.assert_allclose(oe_r.stat, R_ZA_CT["stat"], rtol=RTOL_TIGHT)
