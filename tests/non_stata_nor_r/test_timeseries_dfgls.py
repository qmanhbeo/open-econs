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

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from arch.unitroot import DFGLS as ArchDFGLS

import open_econs as oe

REPO_ROOT = Path(__file__).resolve().parents[2]
UR_INPUT = REPO_ROOT / "tests" / "r" / "fixtures" / "inputs" / "ur_input.csv"


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
