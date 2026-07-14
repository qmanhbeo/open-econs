import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

import open_econs as oe


def _sim_staggered(tau=2.0, n=200, T=6, seed=0):
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(n), T)
    t = np.tile(np.arange(T), n)
    # Staggered adoption: entities adopt at period 1, 3, or 5 (and some never).
    adopt = rng.choice([1, 3, 5, 99], size=n, p=[0.3, 0.3, 0.3, 0.1])
    treat = np.zeros(n * T)
    for e in range(n):
        for j in range(T):
            idx = e * T + j
            treat[idx] = 1.0 if adopt[e] <= j and adopt[e] < 99 else 0.0
    mu = rng.normal(0, 1, size=n)
    time_fe = rng.normal(0, 0.5, size=T)
    y = mu[ent] + time_fe[t] + tau * treat + rng.normal(0, 1, size=n * T)
    return pd.DataFrame({"y": y, "entity": ent, "time": t, "treat": treat})


def test_staggered_recovers_constant_att():
    df = _sim_staggered(tau=2.0, seed=0)
    r = oe.did_cs(df, y="y", entity="entity", time="time", treatment="treat")
    assert np.isclose(r.att, 2.0, atol=0.2)
    assert r.att_p < 0.05
    assert set(r.cohorts) == {1, 3, 5}
    # Every post-treatment group-time ATT should be near the true effect
    # (individual cells have SE ~0.2, so allow a little slack).
    assert np.allclose(r.att_group_time["att"], 2.0, atol=0.5)


def test_staggered_event_study_positive_leads():
    df = _sim_staggered(tau=2.0, seed=1)
    r = oe.did_cs(df, y="y", entity="entity", time="time", treatment="treat")
    assert (r.event_study["lead"] >= 0).all()
    assert np.isclose(r.event_study["att"].mean(), 2.0, atol=0.3)


def test_staggered_matches_simple_did_when_single_cohort():
    # All treated at the same time -> reduces to ordinary two-period DiD.
    rng = np.random.default_rng(3)
    n = 200
    T = 4
    ent = np.repeat(np.arange(n), T)
    t = np.tile(np.arange(T), n)
    treat = ((t >= 2) & (ent % 2 == 0)).astype(float)
    mu = rng.normal(0, 1, size=n)
    y = mu[ent] + 1.5 * treat + rng.normal(0, 1, size=n * T)
    df = pd.DataFrame({"y": y, "entity": ent, "time": t, "treat": treat})
    r = oe.did_cs(df, y="y", entity="entity", time="time", treatment="treat")
    # The treated cohort adopts at t=2; its post ATT should be ~1.5.
    post = r.att_group_time[r.att_group_time["time"] >= 2]
    assert np.isclose(post["att"].mean(), 1.5, atol=0.3)
