"""DF-GLS (ERS) unit-root test -- internal backend-identity regression test.

Placement rationale (rule 7)
-----------------------------
This test lives in ``non_stata_nor_r/`` (NOT ``stata/`` or ``r/``) because the
asserted quantity is an **internal OE-vs-arch computational identity**: OE's
``dfgls`` wraps ``arch.unitroot.DFGLS`` directly, so feeding both the same args
must yield a byte-identical statistic.  There is no genuine *external-engine*
anchor for an equality assertion here:

* Stata ``dfgls`` uses Ng-Perron sequential-t / SIC / MAIC lag selection,
  whereas OE (matching arch) uses AIC on the GLS-detrended series.  The lag-
  selection *method* differs by design, so Stata's reported statistic is NOT
  asserted equal to OE's (judgment call 2).  The GLS detrending (ERS cbar) and
  the max-lag ceiling are shared with arch and are covered by the identity
  test below.

The Ng-Perron lag-selection port is tracked in FUTURE_WORK.md; until then the
statistic is explicitly NOT cross-checked against Stata (standing rule 2: we
do not relax a tolerance to force a match).

This is a regression guard: it pins the wrapper so a future refactor that
accidentally diverges from the arch backend is caught immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from arch.unitroot import DFGLS as ArchDFGLS

import open_econs as oe

REPO_ROOT = Path(__file__).resolve().parents[2]
UR_INPUT = REPO_ROOT / "tests" / "r" / "fixtures" / "inputs" / "ur_input.csv"

# The TS-2 xfail below asserts OE's AIC-selected DF-GLS statistic against the
# Stata ``dfgls`` reference at Stata's Ng-Perron SIC/MAIC-selected lag.  The
# committed fixture ``tests/stata/fixtures/expected/ur_dfgls_c.dta`` is read via
# the shared Stata runner (rule 7); we add the ``tests/stata`` package to the
# path so the runner imports cleanly from this ``non_stata_nor_r`` module.
sys.path.insert(0, str(REPO_ROOT / "tests"))
from stata.stata_runner import read_stata  # noqa: E402

S_DFGLS_C = read_stata("ur_dfgls_c")


def _y() -> np.ndarray:
    return pd.read_csv(UR_INPUT)["y"].astype(float).to_numpy()


@pytest.mark.parametrize("trend", ["c", "ct"])
class TestDFGLSArchIdentity:
    """OE ``dfgls`` must reproduce ``arch.unitroot.DFGLS`` exactly."""

    def test_statistic_identical(self, trend):
        y = _y()
        oe_r = oe.dfgls(y, trend=trend, method="aic")
        arch_r = ArchDFGLS(y, lags=None, trend=trend, max_lags=None, method="aic")
        npt.assert_allclose(oe_r.stat, float(arch_r.stat), rtol=0, atol=1e-12)

    def test_lags_identical(self, trend):
        y = _y()
        oe_r = oe.dfgls(y, trend=trend, method="aic")
        arch_r = ArchDFGLS(y, lags=None, trend=trend, max_lags=None, method="aic")
        assert oe_r.lags == int(arch_r.lags)

    def test_cv_vintage_label(self, trend):
        # Same-backend regression guard only (arch's own DF-GLS simulation CV).
        oe_r = oe.dfgls(_y(), trend=trend, method="aic")
        assert "MacKinnon" in oe_r.cv_vintage


class TestDFGLSStataLagSelectionGap:
    """FUTURE_WORK TS-2 (line ~273): OE ``dfgls`` uses arch's AIC lag selection;
    Stata ``dfgls`` uses Ng-Perron sequential-t / SIC / MAIC.

    On this 200-obs series the divergence is concrete and source-verified
    (Stata output captured in ``tests/stata/generate-fixtures/ur_dfgls_c.do``):

    * OE (AIC)            -> selected lag = 0, DF-GLS mu = -1.2045709
    * Stata Ng-Perron SIC -> selected lag = 1  (``r(siclag)`` = 1)
    * Stata Ng-Perron MAIC-> selected lag = 1  (``r(maiclag)`` = 1)
    * Stata DF-GLS mu at its SIC/MAIC lag (=1) = -1.1362432

    (Stata's *seq-t* rule happens to pick 0 here and so agrees with AIC to
    ~1e-9 -- that coincidence is NOT the parity claim; SIC/MAIC are Stata's
    reported optima and they differ.)  Because the lag-selection *method*
    differs by design, OE's statistic cannot equal Stata's SIC/MAIC statistic.
    These xfail tests capture the REAL divergence (structural lag-count gap +
    the statistic gap) so the future ``method="ng-perron"`` port lands as an
    xpass.  Standing rule 2: no tolerance is loosened -- the arch-identity test
    above stays exact.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="TS-2 (FUTURE_WORK line ~273): OE dfgls uses arch AIC (lag=0);"
        " Stata dfgls Ng-Perron SIC/MAIC selects lag=1. The AIC-selected lag"
        " count therefore does not equal Stata's SIC/MAIC lag until the "
        "Ng-Perron port lands.",
    )
    def test_selected_lag_matches_stata_ngperron(self):
        oe_r = oe.dfgls(_y(), trend="c", method="aic")
        assert oe_r.lags == int(S_DFGLS_C["siclag"])

    @pytest.mark.xfail(
        strict=True,
        reason="TS-2 (FUTURE_WORK line ~273): OE AIC-lag DF-GLS mu (-1.2045709)"
        " != Stata SIC/MAIC-lag DF-GLS mu (-1.1362432); the two lag-selection"
        " methods pick different lags so the statistics differ by ~6.8e-2 "
        ">> 1e-6 until the Ng-Perron port lands.",
    )
    def test_statistic_matches_stata_ngperron(self):
        oe_r = oe.dfgls(_y(), trend="c", method="aic")
        npt.assert_allclose(oe_r.stat, S_DFGLS_C["stat_siclag"], atol=1e-6)
