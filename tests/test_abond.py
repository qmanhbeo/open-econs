import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import open_econs as oe


def _simulate_dynamic_panel(
    n=120, T=8, ar=0.6, beta=0.5, sigma_u=1.0, sigma_e=0.4, seed=0,
):
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(n), T)
    t = np.tile(np.arange(T), n)
    mu = rng.normal(0, sigma_u, size=n)
    # Persistent x so lagged levels are strong GMM instruments (the canonical
    # difference-GMM setup); iid x would be weakly identified and bias the
    # coefficient, exactly as in Stata's xtabond2.
    x = np.zeros(n * T)
    for e in range(n):
        for j in range(T):
            idx = e * T + j
            prev = x[e * T + j - 1] if j > 0 else 0.0
            x[idx] = 0.7 * prev + rng.normal(0, 1)
    y = np.zeros(n * T)
    for e in range(n):
        for j in range(T):
            idx = e * T + j
            lag = y[e * T + j - 1] if j > 0 else 0.0
            y[idx] = ar * lag + beta * x[idx] + mu[e] + rng.normal(0, sigma_e)
    return pd.DataFrame({"y": y, "x": x, "entity": ent, "time": t})


def test_abond_recovers_dynamic_coefficients():
    df = _simulate_dynamic_panel()
    r = oe.abond("y ~ x", data=df, entity="entity", time="time")
    assert np.isclose(r.coefficients["L1.y"], 0.6, atol=0.08)
    assert np.isclose(r.coefficients["x"], 0.5, atol=0.08)
    assert r.step == "two-step"


def test_abond_two_step_larger_than_one_step_se():
    df = _simulate_dynamic_panel()
    r2 = oe.abond("y ~ x", data=df, entity="entity", time="time", step="two-step")
    r1 = oe.abond("y ~ x", data=df, entity="entity", time="time", step="one-step")
    assert (r2.std_errors >= r1.std_errors * 0.5).all()


def test_abond_internal_consistency():
    df = _simulate_dynamic_panel()
    r = oe.abond("y ~ x", data=df, entity="entity", time="time", lags=2)
    assert list(r.coefficients.index) == ["L1.y", "L2.y", "x"]
    assert r.n_entities == 120
    assert r.n_obs == 120 * (8 - 3)  # 5 differenced equations per entity
    assert r.n_instruments == 6 * (1 + 1)
    assert r.hansen_j_dof == r.n_instruments - 3
    assert r.ar1_pvalue <= 0.05
    assert r.ar2_pvalue > 0.05


def test_abond_valid_hansen_when_underidentified():
    df = _simulate_dynamic_panel(n=200, T=12)
    r = oe.abond("y ~ x", data=df, entity="entity", time="time")
    assert 0.0 <= r.hansen_j_pvalue <= 1.0


def test_abond_edge_cases():
    df = _simulate_dynamic_panel()
    short = df[df["time"] < 2].copy()
    with pytest.raises(ValueError):
        oe.abond("y ~ x", data=short, entity="entity", time="time")
    with pytest.raises(ValueError):
        oe.abond("y ~ x", data=df, entity="entity", time="time", step="three-step")
    with pytest.raises(ValueError):
        oe.abond("y ~ x", data=df, entity="entity", time="time", lags=0)


def test_abond_panelcontext_and_context_wrappers():
    df = _simulate_dynamic_panel()
    pc = oe.PanelContext(df, entity="entity", time="time")
    r1 = pc.abond("y ~ x")
    r2 = oe.Context(df).abond("y ~ x", entity="entity", time="time")
    assert np.allclose(r1.coefficients.values, r2.coefficients.values)


def test_abond_tiny_panel_finite():
    rng = np.random.default_rng(7)
    T = 4
    y = np.zeros(T)
    x = rng.normal(size=T)
    y[0] = rng.normal()
    for j in range(1, T):
        y[j] = 0.5 * y[j - 1] + x[j] + rng.normal(0, 0.1)
    df = pd.DataFrame({"y": y, "x": x, "entity": 0, "time": np.arange(T)})
    r = oe.abond("y ~ x", data=df, entity="entity", time="time", step="one-step")
    assert np.all(np.isfinite(r.coefficients.values))
    assert np.all(np.isfinite(r.std_errors.values))
    assert r.n_entities == 1
