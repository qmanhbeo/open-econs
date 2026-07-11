"""Public-API tests for the linear GMM estimator ``oe.gmm`` / ``GMMResult``.

Three families, each anchored to an *independent* reference computation (never a
fabricated number):

1. Exactly-identified identity-weight GMM must reproduce textbook 2SLS -- we use
   ``oe.iv`` (linearmodels) as the independent reference for both coefficients
   and standard errors.
2. Overidentified GMM: the two-step estimator reweights and is *not* the 2SLS
   point estimate; the Hansen J overidentification test must have correct size
   (rejection rate ~ nominal alpha under valid instruments) and power (rejects
   when an instrument is invalid).
3. ``cov_type="cluster"`` must produce *larger* standard errors than
   ``cov_type="robust"`` on data with genuine within-cluster error correlation
   -- proving clustering is actually doing something, not silently falling back.
"""

import numpy as np
import pandas as pd
import pytest

import open_econs as oe


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def exactly_identified():
    """1 endogenous regressor, 1 instrument, intercept -- L == p."""
    rng = np.random.default_rng(20240711)
    n = 400
    z = rng.normal(0, 1, n)
    x = 0.5 + 0.8 * z + rng.normal(0, 0.5, n)
    y = 1.0 + 0.5 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


@pytest.fixture
def overidentified():
    """1 endogenous regressor, 2 instruments, intercept -- L = 3 > p = 2."""
    rng = np.random.default_rng(101)
    n = 600
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    x = 0.5 + 0.5 * z1 + 0.5 * z2 + rng.normal(0, 1, n)  # relevant instruments
    u = rng.normal(0, 1, n)                              # valid: u ⊥ (z1, z2)
    y = 1.0 + 1.0 * x + u
    return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})


@pytest.fixture
def clustered():
    """Genuine within-cluster error correlation; x shares a cluster component
    with a valid instrument z so identification leans on between-cluster
    variation (where the cluster random effect lives)."""
    rng = np.random.default_rng(7)
    n_cl, m = 40, 25
    n = n_cl * m
    cid = np.repeat(np.arange(n_cl), m)
    b = rng.normal(0, 1, n_cl)   # cluster component shared by x and z
    a = rng.normal(0, 1, n_cl)   # cluster random effect in the error
    b_i, a_i = b[cid], a[cid]
    x = b_i + rng.normal(0, 1, n)
    z = b_i + rng.normal(0, 1, n)
    u = a_i + rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x + u
    return pd.DataFrame({"y": y, "x": x, "z": z, "cid": cid})


# ── 1. exactly-identified == 2SLS ───────────────────────────────────────────

class TestGmmExactlyIdentifiedParity:
    def test_coefficients_match_iv(self, exactly_identified):
        df = exactly_identified
        iv = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="robust")
        for step in ("onestep", "twostep"):
            r = oe.gmm("y ~ 1 | x ~ z", data=df, step=step, cov_type="robust")
            # point estimate is step-invariant and equals 2SLS
            assert np.allclose(r.coefficients.values, iv.coefficients.values, atol=1e-8)

    def test_standard_errors_match_iv(self, exactly_identified):
        df = exactly_identified
        iv = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="robust")
        # generic-default identity-weight GMM robust SE == 2SLS robust SE
        for step in ("onestep", "twostep"):
            r = oe.gmm("y ~ 1 | x ~ z", data=df, step=step, cov_type="robust")
            assert np.allclose(
                r.std_errors.values, iv.std_errors.values, rtol=1e-8, atol=1e-10
            )

    def test_hansen_j_undefined_when_exactly_identified(self, exactly_identified):
        df = exactly_identified
        r = oe.gmm("y ~ 1 | x ~ z", data=df, step="twostep", cov_type="robust")
        assert r.hansen_j_dof == 0
        assert np.isnan(r.hansen_j_pvalue)


# ── 2. overidentified: efficient GMM != 2SLS, Hansen J sane ─────────────────

class TestGmmOveridentified:
    def test_twostep_reweights_away_from_2sls(self, overidentified):
        df = overidentified
        iv = oe.iv("y ~ 1 | x ~ z1 + z2", data=df, cov_type="robust")
        one = oe.gmm("y ~ 1 | x ~ z1 + z2", data=df, step="onestep", cov_type="robust")
        two = oe.gmm("y ~ 1 | x ~ z1 + z2", data=df, step="twostep", cov_type="robust")

        # one-step identity-weight GMM IS 2SLS; two-step is the efficient reweight
        assert np.allclose(one.coefficients.values, iv.coefficients.values, atol=1e-8)
        diff = float(np.max(np.abs(two.coefficients.values - iv.coefficients.values)))
        # the efficient estimator must move away from 2SLS (measurably, not just
        # floating-point roundoff) -- efficient GMM reweights the moment conditions
        assert diff > 1e-6

    def test_hansen_j_size_and_power(self, overidentified):
        """Monte Carlo: under valid instruments Hansen J rejects ~alpha (size);
        under an invalid instrument it rejects far more often (power)."""
        rng = np.random.default_rng(2024)
        R = 500
        n = 300
        alpha = 0.05
        rej_valid, rej_invalid = 0, 0
        formula = "y ~ 1 | x ~ z1 + z2"
        for _ in range(R):
            z1 = rng.normal(0, 1, n)
            z2 = rng.normal(0, 1, n)
            x = 0.5 + 0.5 * z1 + 0.5 * z2 + rng.normal(0, 1, n)
            # valid: error orthogonal to both instruments
            u_valid = rng.normal(0, 1, n)
            y_v = 1.0 + 1.0 * x + u_valid
            df_v = pd.DataFrame({"y": y_v, "x": x, "z1": z1, "z2": z2})
            p_v = oe.gmm(formula, data=df_v, step="twostep", cov_type="robust").hansen_j_pvalue
            if p_v < alpha:
                rej_valid += 1
            # invalid: second instrument is correlated with the error
            u_invalid = 0.8 * z2 + rng.normal(0, 1, n)
            y_i = 1.0 + 1.0 * x + u_invalid
            df_i = pd.DataFrame({"y": y_i, "x": x, "z1": z1, "z2": z2})
            p_i = oe.gmm(formula, data=df_i, step="twostep", cov_type="robust").hansen_j_pvalue
            if p_i < alpha:
                rej_invalid += 1

        size = rej_valid / R
        power = rej_invalid / R
        # size must be near nominal alpha; power must be substantially higher
        assert 0.02 <= size <= 0.10, f"Hansen J size out of range: {size}"
        assert power > 0.5, f"Hansen J power too low: {power}"


# ── 3. cluster vs robust ────────────────────────────────────────────────────

class TestGmmClusterVsRobust:
    def test_cluster_se_larger_than_robust(self, clustered):
        df = clustered
        robust = oe.gmm("y ~ 1 | x ~ z", data=df, step="twostep", cov_type="robust")
        cluster = oe.gmm(
            "y ~ 1 | x ~ z", data=df, step="twostep", cov_type="cluster", cluster="cid"
        )
        # coefficients are unchanged by the covariance choice
        assert np.allclose(cluster.coefficients.values, robust.coefficients.values, atol=1e-10)
        # clustering must inflate the SE (genuine within-cluster correlation)
        assert cluster.std_errors["x"] > robust.std_errors["x"]
        assert cluster.std_errors["x"] / robust.std_errors["x"] > 1.2

    def test_cluster_requires_cluster_arg(self, clustered):
        df = clustered
        with pytest.raises(ValueError):
            oe.gmm("y ~ 1 | x ~ z", data=df, cov_type="cluster")

    def test_cluster_arg_unused_with_robust_raises(self, clustered):
        df = clustered
        with pytest.raises(ValueError):
            oe.gmm("y ~ 1 | x ~ z", data=df, cov_type="robust", cluster="cid")


# ── 4. result-class interface ──────────────────────────────────────────────

class TestGmmResultInterface:
    def test_tidy_summary_vcov_export(self, exactly_identified, tmp_path):
        df = exactly_identified
        r = oe.gmm("y ~ 1 | x ~ z", data=df, step="twostep", cov_type="robust")

        t = r.tidy()
        assert list(t.columns) == [
            "Variable", "Coef", "Std Err", "z", "P>|z|", "0.025", "0.975",
        ]
        assert list(t["Variable"]) == ["Intercept", "x"]

        assert isinstance(r.summary(), str)
        assert "Hansen J" in r.summary()

        v = r.vcov()
        assert v.shape == (2, 2)
        assert np.allclose(np.diag(v), r.std_errors.values ** 2)

        csv = tmp_path / "gmm.csv"
        r.export(str(csv))
        assert csv.exists()
        jsonp = tmp_path / "gmm.json"
        r.export(str(jsonp))
        assert jsonp.exists()

    def test_immutable(self, exactly_identified):
        df = exactly_identified
        r = oe.gmm("y ~ 1 | x ~ z", data=df, step="twostep", cov_type="robust")
        with pytest.raises(AttributeError):
            r.coefficients = pd.Series([1.0, 2.0], index=["Intercept", "x"])
