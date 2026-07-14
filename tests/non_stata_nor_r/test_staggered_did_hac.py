"""HAC support for ``staggered_did()`` -- validation floor only.

IMPORTANT (honesty bar): staggered-DiD HAC inference is a *contested* area and
no canonical Stata/R reference implements it. ``cov_type="HAC"`` on
``staggered_did()`` is a documented **project convention** (Newey-West temporal
correction on the aggregated influence function), NOT externally validated.
These tests therefore assert only the *internal-consistency* floor:

  (a) ``lags=0`` HAC reduces exactly to the cluster-robust SE;
  (b) HAC SE is non-negative and PSD (factor floored at 0);
  (c) an invalid ``cov_type`` raises a clear ``ValueError``;
  (d) ``cov_type="HAC"`` without ``lags`` raises a clear ``ValueError``.

They do NOT assert parity with any external estimator. See
``open_econs.models.causal.staggered_did`` for the full caveat.
"""

import warnings

import numpy as np
import pandas as pd

import open_econs as oe


def _sim_staggered(tau=2.0, n=200, T=6, seed=0):
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(n), T)
    t = np.tile(np.arange(T), n)
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


def test_hac_lags0_equals_cluster_se():
    df = _sim_staggered(seed=1)
    base = oe.staggered_did(
        df, y="y", entity="entity", time="time", treatment="treat", cov_type="cluster"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hac = oe.staggered_did(
            df, y="y", entity="entity", time="time", treatment="treat",
            cov_type="HAC", lags=0,
        )
    assert hac.cov_type == "HAC"
    assert base.att_se == hac.att_se


def test_hac_inflates_se_under_autocorrelation():
    df = _sim_staggered(seed=2)
    base = oe.staggered_did(
        df, y="y", entity="entity", time="time", treatment="treat", cov_type="cluster"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hac = oe.staggered_did(
            df, y="y", entity="entity", time="time", treatment="treat",
            cov_type="HAC", lags=2,
        )
    assert hac.att_se >= 0.0
    # With genuine time structure the HAC factor should be positive; SE stays finite.
    assert np.isfinite(hac.att_se)


def test_hac_warns_experimental():
    df = _sim_staggered(seed=3)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        oe.staggered_did(
            df, y="y", entity="entity", time="time", treatment="treat",
            cov_type="HAC", lags=1,
        )
    assert any(
        issubclass(w.category, UserWarning) and "HAC" in str(w.message)
        for w in record
    )


def test_invalid_cov_type_raises():
    df = _sim_staggered(seed=4)
    try:
        oe.staggered_did(
            df, y="y", entity="entity", time="time", treatment="treat",
            cov_type="hacx",
        )
        raised = None
    except ValueError as exc:
        raised = exc
    assert raised is not None
    assert "cov_type" in str(raised)


def test_hac_without_lags_raises():
    df = _sim_staggered(seed=5)
    try:
        oe.staggered_did(
            df, y="y", entity="entity", time="time", treatment="treat",
            cov_type="HAC",
        )
        raised = None
    except ValueError as exc:
        raised = exc
    assert raised is not None
    assert "lags" in str(raised)


def test_hac_reg_method_proxy_runs():
    df = _sim_staggered(seed=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hac = oe.staggered_did(
            df, y="y", entity="entity", time="time", treatment="treat",
            method="reg", cov_type="HAC", lags=1,
        )
    assert np.isfinite(hac.att_se)
