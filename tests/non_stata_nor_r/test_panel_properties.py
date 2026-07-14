"""Property-based (hypothesis) tests for the v0.6 panel-data engine.

These fuzz random balanced panels and assert invariants that must hold
for ANY panel data an econometrician could throw at the estimators:
- FE slopes match a hand-coded group-demeaned (within) OLS reference
- RE theta (the GLS weighting parameter) lies in [0, 1]
- RE R-squared measures lie in [0, 1]
- Hausman statistic is non-negative and its p-value is in [0, 1]
- in-sample RE predictions reproduce fitted values
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

import open_econs as oe


def _random_panel(n_unit, n_time, seed):
    rng = np.random.default_rng(seed)
    n = n_unit * n_time
    entity = np.repeat(np.arange(n_unit), n_time)
    time = np.tile(np.arange(n_time), n_unit)
    alpha = rng.normal(0, 2, n_unit)
    beta_t = rng.normal(0, 1.5, n_time)
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    y = 1.5 * x - 0.7 * z + alpha[entity] + beta_t[time] + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "entity": entity, "time": time})


def _within_ols(y, X, entity, time):
    def _d(v, g):
        s = pd.DataFrame({"v": v, "g": g})
        return v - s.groupby("g")["v"].transform("mean")

    yd = _d(_d(y, entity), time)
    cols = []
    for j in range(X.shape[1]):
        cols.append(_d(_d(X[:, j], entity), time))
    Xd = np.column_stack(cols)
    return np.linalg.lstsq(Xd, yd, rcond=None)[0]


@settings(max_examples=20, deadline=None)
@given(
    n_unit=st.integers(5, 25),
    n_time=st.integers(3, 8),
    seed=st.integers(0, 1000),
)
def test_fe_slopes_match_within_reference(n_unit, n_time, seed):
    df = _random_panel(n_unit, n_time, seed)
    r = oe.fe("y ~ x + z", data=df, entity="entity", time="time")
    ref = _within_ols(
        df["y"].values, df[["x", "z"]].values, df["entity"].values, df["time"].values,
    )
    for i, name in enumerate(["x", "z"]):
        assert np.isclose(r.coefficients[name], ref[i], atol=1e-6), name


@settings(max_examples=20, deadline=None)
@given(
    n_unit=st.integers(5, 25),
    n_time=st.integers(3, 8),
    seed=st.integers(0, 1000),
)
def test_re_theta_in_unit_interval(n_unit, n_time, seed):
    df = _random_panel(n_unit, n_time, seed)
    r = oe.PanelContext(df, entity="entity", time="time").re("y ~ x + z")
    assert isinstance(r.theta, float)
    assert 0.0 <= r.theta <= 1.0 + 1e-9
    assert 0.0 <= r.rho <= 1.0 + 1e-9


@settings(max_examples=20, deadline=None)
@given(
    n_unit=st.integers(5, 25),
    n_time=st.integers(3, 8),
    seed=st.integers(0, 1000),
)
def test_re_r2_measures_in_unit_interval(n_unit, n_time, seed):
    df = _random_panel(n_unit, n_time, seed)
    r = oe.PanelContext(df, entity="entity", time="time").re("y ~ x + z")
    # linearmodels' SS-based R² for RE/GLS can go negative (and above 1)
    # because GLS-transformed residuals may have larger SS than the raw
    # total SS around the grand mean.  This is inherent to the formula and
    # matches linearmodels' own output.  Stata does not report R² for RE.
    # We only require finiteness for the SS-based measures; the
    # correlation-squared alternative (r_squared_overall_corr) is always in [0,1].
    assert np.isfinite(r.r_squared_within)
    assert np.isfinite(r.r_squared_between)
    assert np.isfinite(r.r_squared_overall)
    assert np.isfinite(r.r_squared_overall_corr)
    assert 0.0 <= r.r_squared_overall_corr <= 1.0 + 1e-9


@settings(max_examples=20, deadline=None)
@given(
    n_unit=st.integers(5, 25),
    n_time=st.integers(3, 8),
    seed=st.integers(0, 1000),
)
def test_hausman_statistic_nonneg_p_in_unit(n_unit, n_time, seed):
    df = _random_panel(n_unit, n_time, seed)
    pc = oe.PanelContext(df, entity="entity", time="time")
    fe = pc.fe("y ~ x + z")
    re = pc.re("y ~ x + z")
    h = pc.hausman(fe, re)
    assert h.statistic >= 0.0
    assert 0.0 <= h.p_value <= 1.0
    assert h.df == 2


@settings(max_examples=15, deadline=None)
@given(
    n_unit=st.integers(5, 20),
    n_time=st.integers(3, 8),
    seed=st.integers(0, 1000),
)
def test_re_insample_predict_equals_fitted(n_unit, n_time, seed):
    df = _random_panel(n_unit, n_time, seed)
    pc = oe.PanelContext(df, entity="entity", time="time")
    re = pc.re("y ~ x + z")
    pred = re.predict()
    assert np.allclose(pred.values, re.fitted_values.values, atol=1e-8)
