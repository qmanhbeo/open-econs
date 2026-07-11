"""Unit tests for the McCrary / Cattaneo-Jansson-Ma density (manipulation) test.

These do not require Stata.  They check the *behavior* of the test: it must
reject a deliberately manipulated running variable, fail to reject a smooth
one, and degrade gracefully on degenerate inputs.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import open_econs as oe
from open_econs.models.causal.rdd import _RDENSITY, DensityTestResult


def _manipulated(seed: int = 123) -> pd.DataFrame:
    """Running variable with a clear pile-up just above the cutoff (cutoff 0)."""
    rng = np.random.default_rng(seed)
    x_below = rng.uniform(-1.0, 0.0, 1500)
    x_pile = rng.uniform(0.0, 0.04, 600)       # extra mass right at the cutoff
    x_above = rng.uniform(0.15, 1.0, 900)      # gap then continuous tail
    x = np.concatenate([x_below, x_pile, x_above])
    return pd.DataFrame({"x": x, "y": x})


def _smooth(seed: int = 5) -> pd.DataFrame:
    """Smooth (uniform) running variable - no manipulation."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, 3000)
    return pd.DataFrame({"x": x, "y": x})


def test_density_rejects_manipulation():
    df = _manipulated()
    for backend in ("rddensity", "builtin"):
        if backend == "rddensity" and not _RDENSITY:
            pytest.skip("rddensity not installed")
        r = oe.density_test(df, "x", 0.0, backend=backend)
        # Unambiguous rejection (p well below 0.01 - robust to numerical noise).
        assert r.p_value < 0.01
        # Density is higher just above the cutoff (positive theta).
        assert r.theta > 0
        assert r.fhat_right > r.fhat_left
        assert isinstance(r, DensityTestResult)


def test_density_no_reject_smooth():
    df = _smooth()
    res = {}
    for backend in ("rddensity", "builtin"):
        if backend == "rddensity" and not _RDENSITY:
            pytest.skip("rddensity not installed")
        r = oe.density_test(df, "x", 0.0, backend=backend)
        res[backend] = r
        assert r.p_value > 0.05
    # The two paths agree on the conclusion (and the statistics).
    if "rddensity" in res and "builtin" in res:
        assert abs(res["rddensity"].theta - res["builtin"].theta) < 1e-6


def test_density_auto_picks_installed_backend():
    df = _smooth()
    r = oe.density_test(df, "x", 0.0, backend="auto")
    expected = "rddensity" if _RDENSITY else "builtin"
    assert r.backend == expected


def test_density_edge_few_observations():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, 12)
    df = pd.DataFrame({"x": x, "y": x})
    # Must not raise; returns a finite or nan result gracefully.
    r = oe.density_test(df, "x", 0.0, backend="builtin")
    assert isinstance(r, DensityTestResult)


def test_density_edge_cutoff_at_boundary():
    # All observations on one side of the cutoff -> the other side is empty.
    rng = np.random.default_rng(1)
    x = rng.uniform(0.1, 1.0, 500)
    df = pd.DataFrame({"x": x, "y": x})
    r = oe.density_test(df, "x", 0.0, backend="builtin")
    # Left side empty -> no finite two-sided estimate, but no crash.
    assert np.isnan(r.theta) or r.n_left == 0
    assert isinstance(r, DensityTestResult)


def test_rdresult_density_test_delegates():
    df = _smooth()
    rd = oe.rdd(df, "y", "x", 0.0, bandwidth_select="ik", vce="ehw")
    r = rd.density_test(df, backend="builtin")
    assert isinstance(r, DensityTestResult)
    assert r.n_left + r.n_right == len(df)
