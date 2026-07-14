import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import open_econs as oe


def _sim_rdd(tau=3.0, n=2000, cutoff=0.0, seed=0, fuzzy=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=n)
    eps = rng.normal(0, 1, size=n)
    if fuzzy:
        # Treatment jumps at the cutoff but with imperfect compliance.
        # Treatment probability: 0.2 below cutoff, 0.8 above (jump of 0.6).
        p = 0.2 + 0.6 * (x >= cutoff).astype(float)
        treat = (rng.uniform(size=n) < p).astype(float)
    else:
        treat = (x >= cutoff).astype(float)
    y = 1.0 * x + tau * treat + eps
    return pd.DataFrame({"y": y, "x": x, "treat": treat})


def test_rdd_sharp_recovers_discontinuity():
    df = _sim_rdd(tau=3.0, seed=0)
    r = oe.rdd(df, y="y", running="x", cutoff=0.0, bandwidth_select="ik",
               vce="ehw")
    assert np.isclose(r.effect, 3.0, atol=0.5)
    assert r.p_value < 0.05
    assert r.n_left > 0 and r.n_right > 0
    assert r.fuzzy is False


def test_rdd_fuzzy_recovers_discontinuity():
    df = _sim_rdd(tau=3.0, fuzzy=True, seed=1)
    r = oe.rdd(df, y="y", running="x", cutoff=0.0, treatment="treat",
               fuzzy=True, bandwidth_select="ik", vce="ehw")
    assert np.isclose(r.effect, 3.0, atol=0.5)
    assert r.fuzzy is True


def test_rdd_bandwidth_reduces_sample():
    df = _sim_rdd(tau=3.0, seed=2)
    r_wide = oe.rdd(df, y="y", running="x", cutoff=0.0, bandwidth=1.5)
    r_narrow = oe.rdd(df, y="y", running="x", cutoff=0.0, bandwidth=0.4)
    assert r_narrow.n_left + r_narrow.n_right < r_wide.n_left + r_wide.n_right
    assert np.isclose(r_narrow.effect, 3.0, atol=0.3)


def test_rdd_invalid_args():
    df = _sim_rdd(seed=0)
    with pytest.raises(ValueError):
        oe.rdd(df, y="y", running="x", cutoff=0.0, fuzzy=True)  # fuzzy needs treatment
    with pytest.raises(ValueError):
        oe.rdd(df, y="y", running="x", cutoff=0.0, kernel="epanechnikov")
