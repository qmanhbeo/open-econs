"""Cross-consistency test for the identity-weight (W=None) path of the shared GMM core.

The shared solver ``open_econs.models._gmm_core.estimate_gmm`` accepts a one-step
weighting matrix ``W``.  When ``W is None`` it defaults to the identity
(``W_inner = I``), giving ``A1 = (Z'Z)^{-1}`` -- the plain/iid GMM case.

Mathematical anchor: when a GMM system is *exactly identified* (number of
instrument columns ``L`` equals number of regressors ``p``) and weighted by the
identity, the GMM estimator is identical (in point estimates) to linear
2SLS/IV.  open-econs already ships a Stata-validated IV-2SLS estimator
(``oe.iv`` -> linearmodels), which is used here as the *independent* reference --
no fabricated numbers, no .dta fixtures.

This file is TEST-ONLY.  It must not change any estimator logic.
"""

import numpy as np
import pandas as pd
import pytest

import open_econs as oe
from open_econs.models._gmm_core import estimate_gmm


@pytest.fixture
def exactly_identified_2sls():
    """Synthetic exactly-identified 2SLS setup: 1 endogenous + 1 instrument + intercept.

    Regressors X = [intercept, x]  (p = 2).
    Instruments Z = [intercept, z] (L = 2) -> L == p (exactly identified).
    """
    np.random.seed(20240711)
    n = 400
    z = np.random.normal(0, 1, n)
    x = 0.5 + 0.8 * z + np.random.normal(0, 0.5, n)
    y = 1.0 + 0.5 * x + np.random.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})

    Y = y.astype(float)
    X = np.column_stack([np.ones(n), x]).astype(float)  # [intercept, x]
    Z = np.column_stack([np.ones(n), z]).astype(float)  # [intercept, z]
    eq = np.arange(n)  # cross-sectional: each obs its own group
    return df, Y, X, Z, eq, n


class TestGmmCoreIdentityWeight:
    def test_w_none_equals_explicit_identity(self, exactly_identified_2sls):
        """W=None must be exactly the identity-weighted case."""
        _, Y, X, Z, eq, n = exactly_identified_2sls
        out_default = estimate_gmm(Y, X, Z, eq, "one-step", False, None)
        out_identity = estimate_gmm(Y, X, Z, eq, "one-step", False, np.eye(n))
        assert np.allclose(out_default["b"], out_identity["b"], atol=1e-10)
        assert np.allclose(out_default["se"], out_identity["se"], atol=1e-10)

    def test_coefficients_match_iv_2sls(self, exactly_identified_2sls):
        """Exactly-identified identity-weight GMM == 2SLS coefficients (tight)."""
        df, Y, X, Z, eq, _ = exactly_identified_2sls
        iv = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="nonrobust")
        gmm = estimate_gmm(Y, X, Z, eq, "one-step", False, None)
        assert np.allclose(gmm["b"], iv.coefficients.values, atol=1e-8)

    def test_coefficients_invariant_to_step_and_robust(self, exactly_identified_2sls):
        """In the exactly-identified case the GMM point estimate does not depend
        on the weighting step/robustness -- it is always the 2SLS estimate."""
        df, Y, X, Z, eq, _ = exactly_identified_2sls
        iv = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="nonrobust")
        for step in ("one-step", "two-step"):
            for robust in (False, True):
                gmm = estimate_gmm(Y, X, Z, eq, step, robust, None)
                assert np.allclose(gmm["b"], iv.coefficients.values, atol=1e-8)

    def test_se_generic_default_matches_iv_nonrobust(self, exactly_identified_2sls):
        """With the new GENERIC defaults (sig2_scale=1.0, no small-sample
        correction) the identity-weight one-step NON-robust SEs must match
        textbook 2SLS exactly -- the Arellano-Bond leaks are gone, so there is
        no convention gap to encode anymore."""
        df, Y, X, Z, eq, _ = exactly_identified_2sls
        iv = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="nonrobust")
        gmm = estimate_gmm(Y, X, Z, eq, "one-step", False, None)
        assert np.allclose(gmm["se"], iv.std_errors.values, rtol=1e-8, atol=1e-10)

    def test_se_generic_default_matches_iv_robust(self, exactly_identified_2sls):
        """With generic defaults, identity-weight robust (one-step & two-step)
        SEs match linearmodels 'robust' (debiased=False) 2SLS exactly -- the
        small-sample multiplier no longer distorts the shared core."""
        df, Y, X, Z, eq, _ = exactly_identified_2sls
        iv = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="robust")
        for step in ("one-step", "two-step"):
            gmm = estimate_gmm(Y, X, Z, eq, step, True, None)
            assert np.allclose(gmm["se"], iv.std_errors.values, rtol=1e-8, atol=1e-10)

    def test_ab_params_reproduce_ab_convention(self, exactly_identified_2sls):
        """Passing the explicit AB values (sig2_scale=0.5,
        small_sample_correction=True) must reproduce the original AB convention
        gaps vs 2SLS -- proving the decoupling is faithful, not a silent change.

        non-robust one-step: factor (1/sqrt(2)) * sqrt(n/(n-k))
        robust (one/two-step): factor sqrt(n/(n-k))
        """
        df, Y, X, Z, eq, n = exactly_identified_2sls
        k = X.shape[1]
        nonrobust_factor = (1.0 / np.sqrt(2.0)) * np.sqrt(n / (n - k))
        robust_factor = np.sqrt(n / (n - k))
        iv_nr = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="nonrobust")
        iv_r = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="robust")
        gmm_nr = estimate_gmm(
            Y, X, Z, eq, "one-step", False, None,
            sig2_scale=0.5, small_sample_correction=True,
        )
        assert np.allclose(gmm_nr["se"], iv_nr.std_errors.values * nonrobust_factor,
                           rtol=1e-8, atol=1e-10)
        for step in ("one-step", "two-step"):
            gmm_r = estimate_gmm(
                Y, X, Z, eq, step, True, None,
                sig2_scale=0.5, small_sample_correction=True,
            )
            assert np.allclose(gmm_r["se"], iv_r.std_errors.values * robust_factor,
                               rtol=1e-8, atol=1e-10)
