"""Parity tests for Newey-West HAC covariance in ``fe()`` (fixed effects).

The design uses the period-aggregation (Arellano / Driscoll-Kraay) convention:
score contributions ``x_it * e_it`` are summed within each time period across
entities, then a Bartlett-kernel long-run variance is applied across periods.
This is implemented by ``core.cov.newey_west_cov`` with ``cluster = time`` and
matches ``statsmodels.stats.sandwich_covariance.cov_nw_groupsum`` (whose
docstring states it is "Tested against STATA xtscc package").

Two independent references are used:
  * statsmodels ``cov_nw_groupsum`` — runs on every CI push (no R binary).
  * an R ``sandwich``-style manual computation — dual-mode fixture under
    ``tests/r/`` (runs locally when R is installed; reads committed JSON on CI).
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.stats.sandwich_covariance as sw

import open_econs as oe
from open_econs.core.cov import _as_int_labels
from ..r.r_runner import read_r


def _make_panel(n=8, T=10, seed=123):
    """Deterministic balanced panel with FE, AR(1) errors, and regressors."""
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(n), T)
    t = np.tile(np.arange(T), n)
    alpha = rng.normal(0, 2, n)
    beta_t = rng.normal(0, 1.5, T)
    x = rng.normal(0, 1, n * T)
    z = rng.normal(0, 1, n * T)
    u = rng.normal(0, 1, n * T)
    e = np.zeros(n * T)
    for i in range(1, n * T):
        e[i] = 0.5 * e[i - 1] + u[i]
    y = 1.5 * x - 0.7 * z + alpha[ent] + beta_t[t] + e
    return pd.DataFrame({"y": y, "x": x, "z": z, "entity": ent, "time": t})


def _two_way_demean(v, ent, t):
    s = pd.DataFrame({"v": np.asarray(v, float), "e": ent, "t": t})
    g = s.groupby("e")["v"].transform("mean")
    g = g + s.groupby("t")["v"].transform("mean") - s["v"].mean()
    return np.asarray(v, float) - g.values


class TestFeHacStatsmodels:
    def test_fe_hac_raw_v_matches_cov_nw_groupsum(self):
        df = _make_panel()
        r = oe.fe("y ~ x + z", data=df, entity="entity", time="time",
                  cov_type="HAC", lags=2)
        yd = _two_way_demean(df["y"].values, df["entity"].values, df["time"].values)
        Xd = np.column_stack([
            _two_way_demean(df[c].values, df["entity"].values, df["time"].values)
            for c in ("x", "z")
        ])
        fit = sm.OLS(yd, Xd).fit()
        time_labels = _as_int_labels(df["time"].values)
        ref_V = sw.cov_nw_groupsum(fit, nlags=2, time=time_labels, use_correction=0)
        # Undo the FE df_resid scaling fe() applies so we compare the raw HAC V.
        raw_V = r._cov.values * (r.df_resid / (r.nobs - len(r.coefficients)))
        npt.assert_allclose(raw_V, ref_V, rtol=1e-7, atol=1e-8)

    def test_fe_hac_se_matches_cov_nw_groupsum(self):
        df = _make_panel()
        r = oe.fe("y ~ x + z", data=df, entity="entity", time="time",
                  cov_type="HAC", lags=2)
        yd = _two_way_demean(df["y"].values, df["entity"].values, df["time"].values)
        Xd = np.column_stack([
            _two_way_demean(df[c].values, df["entity"].values, df["time"].values)
            for c in ("x", "z")
        ])
        fit = sm.OLS(yd, Xd).fit()
        time_labels = _as_int_labels(df["time"].values)
        ref_V = sw.cov_nw_groupsum(fit, nlags=2, time=time_labels, use_correction=0)
        raw_V = r._cov.values * (r.df_resid / (r.nobs - len(r.coefficients)))
        npt.assert_allclose(np.sqrt(np.diag(raw_V)), np.sqrt(np.diag(ref_V)),
                            rtol=1e-7, atol=1e-8)

    def test_fe_hac_time_only_fe_works(self):
        # time required for HAC; entity may be omitted (one-way time FE + HAC).
        df = _make_panel()
        r = oe.fe("y ~ x + z", data=df, time="time", cov_type="HAC", lags=1)
        assert r.cov_type == "HAC(1)"
        assert np.all(np.isfinite(r.std_errors.values))


class TestFeHacAdjust:
    def test_hac_adjust_scales_se_by_n_over_nk(self):
        df = _make_panel()
        r0 = oe.fe("y ~ x + z", data=df, entity="entity", time="time",
                   cov_type="HAC", lags=2, hac_adjust=False)
        r1 = oe.fe("y ~ x + z", data=df, entity="entity", time="time",
                   cov_type="HAC", lags=2, hac_adjust=True)
        nobs = len(df)
        k = 2  # x, z (intercept dropped after demeaning)
        factor = np.sqrt(nobs / (nobs - k))
        npt.assert_allclose(r1.std_errors.values / r0.std_errors.values, factor,
                            rtol=1e-10)


class TestFeHacValidation:
    def test_requires_lags(self):
        df = _make_panel()
        try:
            oe.fe("y ~ x + z", data=df, entity="entity", time="time",
                  cov_type="HAC")
        except ValueError as e:
            assert "lags" in str(e)
        else:
            raise AssertionError("expected ValueError for missing lags")

    def test_requires_time(self):
        df = _make_panel()
        try:
            oe.fe("y ~ x + z", data=df, entity="entity", cov_type="HAC", lags=2)
        except ValueError as e:
            assert "time" in str(e)
        else:
            raise AssertionError("expected ValueError for missing time")


class TestFeHacRParity:
    @pytest.mark.r
    def test_fe_hac_matches_r(self):
        df = _make_panel()
        r = oe.fe("y ~ x + z", data=df, entity="entity", time="time",
                  cov_type="HAC", lags=2)
        raw_V = r._cov.values * (r.df_resid / (r.nobs - len(r.coefficients)))

        ref = read_r("fe_hac_parity")
        ref_V = np.array(ref["cov"])
        ref_coef = np.array(ref["coefficients"])
        ref_se = np.array(ref["std_errors"])

        npt.assert_allclose(r.coefficients.values, ref_coef, rtol=1e-6)
        npt.assert_allclose(np.sqrt(np.diag(raw_V)), ref_se, rtol=1e-6)
        npt.assert_allclose(raw_V, ref_V, rtol=1e-6, atol=1e-8)
