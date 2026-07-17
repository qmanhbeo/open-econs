"""Tests for psm() — Abadie-Imbens (2006, 2012) 1:1 PSM with replacement."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree as _cKDTree

from open_econs.models.causal.balance import balance as _balance
from open_econs.models.causal.psm import (
    _compute_ai_variance,
    _compute_local_cov,
    _count_matches,
    _nearest_neighbor_match_with_replacement,
    _compute_propensity_scores,
    _opposite_treatment_matching,
    _within_treatment_matching,
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


# ── Vectorization determinism (rule 19: parity-preserving refactor) ──
# The batched cKDTree / padded-tensor vectorization must reproduce the scalar
# per-unit loops EXACTLY (atol=0, rtol=0), not merely within 1e-6. These tests
# pin that contract so a future reduction-order change is caught immediately.


def _scalar_within_treatment_matching(ps, treatment, n_neighbors):
    """Reference scalar implementation mirroring the pre-vectorization loop."""
    is_treat = treatment == 1
    treat_idx = np.where(is_treat)[0]
    control_idx = np.where(~is_treat)[0]
    n = len(ps)
    tt = _cKDTree(ps[treat_idx].reshape(-1, 1))
    tc = _cKDTree(ps[control_idx].reshape(-1, 1))
    out = [np.array([], dtype=int) for _ in range(n)]
    for i in treat_idx:
        k = min(n_neighbors + 1, len(treat_idx))
        _, idx = tt.query(ps[i].reshape(1, -1), k=k)
        out[i] = treat_idx[idx[0][:n_neighbors]]
    for i in control_idx:
        k = min(n_neighbors + 1, len(control_idx))
        _, idx = tc.query(ps[i].reshape(1, -1), k=k)
        out[i] = control_idx[idx[0][:n_neighbors]]
    return out


def _scalar_opposite_treatment_matching(ps, treatment, n_neighbors):
    """Reference scalar implementation mirroring the pre-vectorization loop."""
    is_treat = treatment == 1
    treat_idx = np.where(is_treat)[0]
    control_idx = np.where(~is_treat)[0]
    out = [np.array([], dtype=int) for _ in range(len(ps))]
    tree_t = _cKDTree(ps[treat_idx].reshape(-1, 1))
    tree_c = _cKDTree(ps[control_idx].reshape(-1, 1))
    for i in treat_idx:
        k = min(n_neighbors, len(control_idx))
        _, idx = tree_c.query(ps[i].reshape(1, -1), k=k)
        out[i] = control_idx[idx[0]]
    for i in control_idx:
        k = min(n_neighbors, len(treat_idx))
        _, idx = tree_t.query(ps[i].reshape(1, -1), k=k)
        out[i] = treat_idx[idx[0]]
    return out


@pytest.mark.parametrize("h", [2, 5, 10])
def test_psm_within_matching_bit_identical_to_scalar(df, h):
    ps, _, _ = _compute_propensity_scores(df, "t", ["x1", "x2"])
    t = df["t"].values
    vec = _within_treatment_matching(ps, t, h)
    ref = _scalar_within_treatment_matching(ps, t, h)
    assert [a.tolist() for a in vec] == [a.tolist() for a in ref]


@pytest.mark.parametrize("h", [2, 5, 10])
def test_psm_opposite_matching_bit_identical_to_scalar(df, h):
    ps, _, _ = _compute_propensity_scores(df, "t", ["x1", "x2"])
    t = df["t"].values
    vec = _opposite_treatment_matching(ps, t, h)
    ref = _scalar_opposite_treatment_matching(ps, t, h)
    assert [a.tolist() for a in vec] == [a.tolist() for a in ref]


def test_psm_c_tau_vectorization_bit_identical(df):
    """The padded-tensor c_tau must equal the per-unit scalar _compute_local_cov loop."""
    ps, z, Vg = _compute_propensity_scores(df, "t", ["x1", "x2"])
    t = df["t"].values.astype(int)
    y = df["y"].values
    pairs = _nearest_neighbor_match_with_replacement(ps, t, 1.0)
    tau = float(np.nanmean([(2 * t[i] - 1) * (y[i] - y[pairs[i]])
                            for i in range(len(y)) if i in pairs]))
    h = 10
    # Vectorized path.
    V_vec = _compute_ai_variance(y, t, ps, pairs, tau, z, Vg, h=h)

    # Reference scalar path (re-derives c_tau via the per-unit loop).
    n = len(y)
    K = _count_matches(pairs, n)
    psi = np.full(n, np.nan)
    for i in range(n):
        if i in pairs:
            psi[i] = (2 * t[i] - 1) * (y[i] - y[pairs[i]])
    dev = psi - tau
    V_base = np.nansum(dev ** 2 + 0.0) / (n ** 2)  # placeholder; recompute exactly below
    wt = _within_treatment_matching(ps, t, h)
    ot = _opposite_treatment_matching(ps, t, h)
    f_deriv = ps * (1.0 - ps)
    c_tau = np.zeros(z.shape[1])
    for i in range(n):
        if i not in pairs:
            continue
        t_i = t[i]
        p_i = ps[i]
        nb_y1 = wt[i] if t_i == 1 else ot[i]
        nb_y0 = wt[i] if t_i == 0 else ot[i]
        cy1 = _compute_local_cov(z, y, nb_y1) if len(nb_y1) >= 2 else np.zeros(z.shape[1])
        cy0 = _compute_local_cov(z, y, nb_y0) if len(nb_y0) >= 2 else np.zeros(z.shape[1])
        c_tau += f_deriv[i] * (cy1 / p_i + cy0 / (1.0 - p_i))
    c_tau /= n
    V_adj_ref = c_tau @ Vg @ c_tau
    # Base variance (recomputed exactly per the module formula, includes xi2).
    xi2 = np.zeros(n)
    for i in range(n):
        nb = wt[i]
        if len(nb) >= 2:
            ym = y[nb].mean()
            xi2[i] = np.sum((y[nb] - ym) ** 2) / (len(nb) - 1)
    Km = K.astype(float)
    V_base = np.nansum(dev ** 2 + xi2 * (Km ** 2 + 2 * Km - Km)) / (n ** 2)
    V_ref = V_base - V_adj_ref

    assert V_vec == V_ref
    # And the SE must still hit the Stata pin.
    se = float(np.sqrt(max(V_vec, 0.0)))
    assert se == pytest.approx(0.08351363, abs=1e-6)
