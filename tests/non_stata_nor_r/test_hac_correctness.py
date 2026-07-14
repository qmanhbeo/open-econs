"""Cross-validation of Newey-West HAC standard errors against statsmodels.

These tests need no Stata/R binary -- they compare open-econs' ``ols(...,
cov_type="HAC")`` against statsmodels' ``OLS.fit(cov_type="HAC")`` (which
delegates to ``statsmodels.stats.sandwich_covariance.cov_hac_simple``).

Convention under test (see ``open_econs/models/linear/ols.py`` and the
Stata/statsmodels source cross-checks in the hand-off report):
  * ``hac_adjust=False`` (default)  == statsmodels default ``use_correction=False``
    == original Newey & West (1987) Bartlett-kernel HAC, no N/(N-K) factor.
  * ``hac_adjust=True``             == statsmodels ``use_correction=True``
    == Stata ``newey`` (which applies N/(N-K) unconditionally).

Both paths are required to match statsmodels at machine precision (rtol=1e-6).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import statsmodels.api as sm

import open_econs as oe


def _ar1_data(n: int = 220, rho: float = 0.55, seed: int = 7) -> pd.DataFrame:
    """Deterministic dataset with AR(1) errors to make HAC non-trivial."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n) + 0.4 * x1
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.standard_normal()
    y = 1.5 + 2.0 * x1 - 1.0 * x2 + e
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "time": t})


@pytest.mark.parametrize("lags", [1, 3, 5])
def test_hac_matches_statsmodels_default(lags: int) -> None:
    """Default HAC (no N/(N-K)) must equal statsmodels use_correction=False."""
    df = _ar1_data()
    oe_r = oe.ols("y ~ x1 + x2", data=df, cov_type="HAC", lags=lags, time="time")
    X = sm.add_constant(df[["x1", "x2"]])
    sm_r = sm.OLS(df["y"].values, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
    )
    npt.assert_allclose(oe_r.coefficients.values, sm_r.params, rtol=1e-9)
    npt.assert_allclose(oe_r.std_errors.values, sm_r.bse, rtol=1e-6)


@pytest.mark.parametrize("lags", [1, 3, 5])
def test_hac_matches_statsmodels_adjust(lags: int) -> None:
    """hac_adjust=True must equal statsmodels use_correction=True (Stata eq.)."""
    df = _ar1_data()
    oe_r = oe.ols(
        "y ~ x1 + x2", data=df, cov_type="HAC", lags=lags,
        time="time", hac_adjust=True,
    )
    X = sm.add_constant(df[["x1", "x2"]])
    sm_r = sm.OLS(df["y"].values, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    npt.assert_allclose(oe_r.coefficients.values, sm_r.params, rtol=1e-9)
    npt.assert_allclose(oe_r.std_errors.values, sm_r.bse, rtol=1e-6)


def test_hac_requires_lags() -> None:
    """Newey-West HAC must error without an explicit lag (Stata-style)."""
    df = _ar1_data()
    with pytest.raises(ValueError):
        oe.ols("y ~ x1 + x2", data=df, cov_type="HAC", time="time")
