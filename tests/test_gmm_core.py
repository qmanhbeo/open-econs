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

    def test_se_nonrobust_convention_gap(self, exactly_identified_2sls):
        """Identity-weight one-step NON-robust SEs differ from textbook 2SLS by a
        precise, explainable factor -- encode it rather than hide it.

        factor = (1 / sqrt(2)) * sqrt(n / (n - k))

        * 1/sqrt(2): the shared core scales sig2 by e1'e1 / (2*n).  That is the
          Arellano-Bond *difference*-GMM normalization (see _gmm_core.py:101,104)
          and is irrelevant to the plain iid case, but it is baked into the
          shared core so it leaks into the W=None path.
        * sqrt(n/(n-k)): the small-sample multiplier the core applies in the
          one-step non-robust branch (_gmm_core.py:177-181).
        """
        df, Y, X, Z, eq, n = exactly_identified_2sls
        k = X.shape[1]
        iv = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="nonrobust")
        gmm = estimate_gmm(Y, X, Z, eq, "one-step", False, None)
        factor = (1.0 / np.sqrt(2.0)) * np.sqrt(n / (n - k))
        assert np.allclose(gmm["se"], iv.std_errors.values * factor, rtol=1e-8, atol=1e-10)

    def test_se_robust_convention_gap(self, exactly_identified_2sls):
        """Robust (one-step and two-step) GMM SEs differ from linearmodels
        'robust' (debiased=False) 2SLS SEs *only* by the small-sample multiplier
        sqrt(n/(n-k)); the AB 1/2 sig2 factor does not enter the robust path.

        This is a genuine convention difference vs a textbook 2SLS reference and
        is asserted exactly (not loosened) so the gap is documented, not hidden.
        """
        df, Y, X, Z, eq, n = exactly_identified_2sls
        k = X.shape[1]
        iv = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="robust")
        factor = np.sqrt(n / (n - k))
        for step in ("one-step", "two-step"):
            gmm = estimate_gmm(Y, X, Z, eq, step, True, None)
            assert np.allclose(gmm["se"], iv.std_errors.values * factor, rtol=1e-8, atol=1e-10)
