"""Tests for psm() — Abadie-Imbens (2006, 2012) 1:1 PSM with replacement."""

import numpy as np
import pandas as pd
import pytest

from open_econs.models.causal.balance import balance as _balance
from open_econs.models.causal.psm import (
    _count_matches,
    _nearest_neighbor_match_with_replacement,
    _compute_propensity_scores,
    psm,
)


FIXTURE = "tests/stata/fixtures/inputs/df_psm.csv"


@pytest.fixture
def df():
    return pd.read_csv(FIXTURE)


def test_psm_ate_matches_stata(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0)
    assert r.effect == pytest.approx(1.5333212, abs=1e-6)


def test_psm_se_nn2_matches_stata(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0, nn=2)
    assert r.se == pytest.approx(0.0765811, abs=1e-6)


def test_psm_se_nn5_matches_stata(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0, nn=5)
    assert r.se == pytest.approx(0.08283881, abs=1e-6)


def test_psm_se_nn10_matches_stata(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0, nn=10)
    assert r.se == pytest.approx(0.08351363, abs=1e-6)


def test_psm_result_shape(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0)
    assert r.n_matched == 500
    assert r.n_treated == 470
    assert r.n_control == 530
    assert isinstance(r.tidy(), pd.DataFrame)
    assert isinstance(r.summary(), str)


def test_psm_caliper_drops_outliers(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=0.01)
    assert r.n_matched < 500
    assert r.effect is not None


def test_psm_common_support(df):
    r1 = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0, common_support=False)
    r2 = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0, common_support=True)
    assert r1.n_matched == 500
    assert r2.n_matched < r1.n_matched


# ── Weights / matched attributes ─────────────────────────────

def test_psm_weights_pattern(df):
    """Treated weight = 1, unused control weight = 0, used control weight >= 1."""
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0)
    t = df["t"].values

    treated_w = r.weights.values[t == 1]
    control_w = r.weights.values[t == 0]

    assert (treated_w == 1.0).all(), "all treated should have weight 1"
    assert (control_w >= 0).all(), "all control weights must be non-negative"
    assert (control_w[control_w > 0] >= 1).all(), "used controls have weight >= 1"
    unused = control_w == 0
    assert unused.any(), "at least one control should be unused"
    assert (r.weights.values >= 0).all()


def test_psm_weights_consistency_with_internal_k(df):
    """Exposed weights equal independently-computed K(i) *from same inputs*."""
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0)

    ps, z, V_gamma = _compute_propensity_scores(df, "t", ["x1", "x2"])
    t = df["t"].values
    pairs = _nearest_neighbor_match_with_replacement(ps, t, 1.0)
    K = _count_matches(pairs, len(df))

    expected_w = np.where(t == 1, 1.0, K.astype(float))
    assert np.array_equal(r.weights.values, expected_w), (
        "exposed weights do not match independently computed K(i) counts"
    )


def test_psm_matched_pattern(df):
    """Matched observations are those with a pair or K(i) > 0."""
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0)
    t = df["t"].values

    # All treated with weight > 0 should be matched
    assert (r.matched.values[t == 1]).all(), "all treated should be matched"
    # Controls with weight > 0 should be matched
    c_matched = r.matched.values[t == 0]
    c_w = r.weights.values[t == 0]
    assert (c_matched == (c_w > 0)).all(), (
        "control matched status should match weight > 0"
    )


def test_psm_result_attributes(df):
    """Smoke-test the new PSMResult attributes."""
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0)
    assert isinstance(r.original_data, pd.DataFrame)
    assert isinstance(r.weights, pd.Series)
    assert isinstance(r.matched, pd.Series)
    assert r.weights.name == "psm_weights"
    assert r.matched.name == "psm_matched"
    assert len(r.weights) == len(df)
    assert len(r.matched) == len(df)


# ── Balance delegation ───────────────────────────────────────

def test_psm_balance_delegation(df):
    """PSMResult.balance() matches direct balance() call with same inputs."""
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0)

    result_bal = r.balance(covariates=["x1", "x2"])

    m = r.matched.values.astype(bool)
    data_m = df.iloc[m].copy()
    data_m["_psm_weights_"] = r.weights.values[m]
    direct_bal = _balance(
        data=data_m,
        treatment="t",
        covariates=["x1", "x2"],
        weights="_psm_weights_",
    )

    pd.testing.assert_frame_equal(result_bal, direct_bal)


def test_psm_balance_shape(df):
    """Smoke test: balance() returns expected columns."""
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0)
    bal = r.balance(covariates=["x1", "x2"])
    assert "SMD" in bal.columns
    assert "Variance Ratio" in bal.columns
    assert len(bal) == 2  # x1 and x2
    assert bal["P>|t|"].is_monotonic_increasing  # sorted by p-value
