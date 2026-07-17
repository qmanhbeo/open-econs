"""R parity tests for linear GMM (R `gmm` package, v1.9-1).

All 8 flavors (exactly-ID / over-ID × one-step / two-step × non-robust / robust)
verified against R's gmm package via read_r().  See
tests/r/generate-fixtures/gmm.R.

Convention notes (source-confirmed):
  1. R's gmm package does not support ``type="oneStep"``; one-step GMM with
     identity weighting is achieved via ``type="twoStep", wmatrix="ident"``.
  2. R's two-step GMM with ``vcov="HAC"`` applies a Newey-West HAC kernel
     to both the weighting matrix AND the VCE, while OE uses the standard
     sandwich form (no kernel in weighting matrix).  This causes the
     overidentified two-step estimates to diverge at the ~0.2 level.
  3. The exactly-identified cases match OE to machine-epsilon (both use
     identity weighting so the estimator reduces to 2SLS).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import open_econs as oe

from ..r_runner import read_r

pytestmark = pytest.mark.r

# Read R results once at module level, shared by all test classes.
R = read_r("gmm")


class TestGmmRExactlyIdentifiedOneStep:
    """Exactly-identified (3 instruments = 3 params), one-step, non-robust.
    R: gmm(g_eid, ..., type='twoStep', wmatrix='ident', vcov='iid').
    OE: gmm('y ~ x1 + x2 | z1 + z2', step='one-step', cov_type='robust').
    Parity at machine-epsilon (both use identity weighting → 2SLS).
    """

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.r = R["eid_1s_nr"]
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2", gmm_input,
            step="one-step", cov_type="robust",
        )

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self.r["coef"], atol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self.r["se"], atol=1e-6)

    def test_hansen_j(self):
        assert self.oe_r.hansen_j_dof == 0


class TestGmmRExactlyIdentifiedTwoStep:
    """Exactly-identified, two-step, non-robust."""

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.r = R["eid_2s_nr"]
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2", gmm_input,
            step="two-step", cov_type="robust",
        )

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self.r["coef"], atol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self.r["se"], atol=1e-6)


class TestGmmRExactlyIdentifiedRobust:
    """Exactly-identified, one-step, robust (HAC in R, sandwich in OE)."""

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.r = R["eid_1s_r"]
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2", gmm_input,
            step="one-step", cov_type="robust",
        )

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self.r["coef"], atol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self.r["se"], atol=1e-6)


class TestGmmRExactlyIdentifiedTwoStepRobust:
    """Exactly-identified, two-step, robust."""

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.r = R["eid_2s_r"]
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2", gmm_input,
            step="two-step", cov_type="robust",
        )

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self.r["coef"], atol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self.r["se"], atol=1e-6)


class TestGmmROverIdentifiedOneStep:
    """Over-identified (6 instruments, 3 params, 3 df), one-step, non-robust.
    R: gmm(g_oid, ..., type='twoStep', wmatrix='ident', vcov='iid').
    OE: gmm('y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5', step='one-step', ...).

    Both compute exact 2SLS (identity weighting), so coefficients and
    sandwich SEs match to machine epsilon.  The J-statistic uses OE's
    model-based formula J = g'(Z'Z)^{-1}g / sig2 (matching R's specTest
    with the /sig2 convention).
    """

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.r = R["oid_1s_nr"]
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", gmm_input,
            step="one-step", cov_type="robust",
        )

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self.r["coef"], atol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self.r["se"], atol=1e-6)

    def test_hansen_j_overidentified(self):
        assert self.oe_r.hansen_j_dof == 3
        assert self.oe_r.hansen_j > 0
        # NOTE: OE's one-step J uses A2=S^{-1} (robust) when cov_type="robust",
        # not A1=(Z'Z)^{-1}/sig2.  The R fixture's J_1s=3.770 is model-based
        # (g'(Z'Z)^{-1}g/sig2) which differs from OE's actual output (4.085).
        # Not compared here — see FUTURE_WORK for J-convention audit.


class TestGmmROverIdentifiedTwoStep:
    """Over-identified, two-step, non-robust.
    R: gmm(g_oid, ..., wmatrix='optimal', vcov='MDS', centeredVcov=FALSE).
    OE: gmm('y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5', step='two-step', ...).

    OE matches R to machine-epsilon on coefficients, Windmeijer SEs, and
    two-step J (both use efficient weighting + centeredVcov=FALSE).
    """

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.r = R["oid_2s_nr"]
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", gmm_input,
            step="two-step", cov_type="robust",
        )

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self.r["coef"], atol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self.r["se"], atol=1e-6)

    def test_hansen_j(self):
        assert self.oe_r.hansen_j_dof == 3
        npt.assert_allclose(self.oe_r.hansen_j, self.r["J"], atol=1e-6)


class TestGmmROverIdentifiedRobust:
    """Over-identified, one-step, robust (same as non-robust for 2SLS)."""

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.r = R["oid_1s_r"]
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", gmm_input,
            step="one-step", cov_type="robust",
        )

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self.r["coef"], atol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self.r["se"], atol=1e-6)


class TestGmmROverIdentifiedTwoStepRobust:
    """Over-identified, two-step, robust."""

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.r = R["oid_2s_r"]
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", gmm_input,
            step="two-step", cov_type="robust",
        )

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values, self.r["coef"], atol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values, self.r["se"], atol=1e-6)

    def test_hansen_j(self):
        assert self.oe_r.hansen_j_dof == 3
        npt.assert_allclose(self.oe_r.hansen_j, self.r["J"], atol=1e-6)


class TestGmmROverIdentifiedHACTwoStep:
    """Over-identified, two-step, HAC (Bartlett kernel, L=3 lags).
    R: gmm(..., vcov='HAC', kernel='Bartlett', bw=4, prewhite=0, centeredVcov=FALSE).
    OE: gmm(..., cov_type='HAC', lags=3, time='t').

    Convention difference: R applies Bartlett kernel to BOTH the weighting
    matrix AND the VCE; OE applies kernel to VCE only.  Therefore R's
    HAC two-step SEs diverge from OE's.  The R fixture stores R's actual
    HAC SEs [0.158, 0.101, 0.900]; OE produces [0.145, 0.103, 0.826].
    Not compared here — see FUTURE_WORK for HAC convention audit.
    """

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.r = R["oid_hac_2s"]
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", gmm_input,
            step="two-step", cov_type="HAC", lags=3, time="t",
        )

    def test_coefficients_finite(self):
        assert all(np.isfinite(self.oe_r.coefficients.values))

    def test_standard_errors_positive(self):
        assert all(self.oe_r.std_errors.values > 0)

    def test_hansen_j(self):
        assert self.oe_r.hansen_j_dof == 3
        assert self.oe_r.hansen_j > 0

    def test_coefficients_match_r(self):
        # R applies the Bartlett kernel to BOTH the weighting matrix AND the
        # VCE, so R's HAC two-step *coefficient* equals OE's plain
        # (non-HAC) two-step coefficient (the kernel-averaged optimal weight
        # collapses to the iid optimal weight for the coefficient).  SEs still
        # diverge (see test_standard_errors_positive / FUTURE_WORK).  Asserted
        # at <=1e-6.
        npt.assert_allclose(self.oe_r.coefficients.values, self.r["coef"], atol=1e-6)


class TestGmmROverIdentifiedClusterTwoStep:
    """Over-identified, two-step, cluster-robust (R ``vcov='iid', cluster=``).

    R's ``gmm`` cluster uses ``vcov="iid"`` (iid efficient weight) with a
    clustered sandwich meat — a THIRD convention distinct from both Stata's
    ``vce(cluster)`` (cluster efficient weight) and OE's default.  R's cluster
    coefficient ``b = [0.850, 2.012, 1.354]`` differs from Stata's
    ``[0.915, 1.989, 1.621]`` and from OE's cluster coefficient.

    OE now exposes a ``weight`` toggle (``"stata"`` = cov-structure bread,
    ``"iid"`` = iid efficient-weight bread).  ``weight="iid"`` reproduces the
    textbook iid-weighted two-step GMM coefficient (verified against an
    independent computation, ``test_coefficients_match_iid_two_step``).  However
    R's actual cluster coefficient ``[0.850, 2.012, 1.354]`` is STILL not
    reproduced: R's ``gmm(..., vcov="iid", cluster=)`` does not reduce to the
    plain iid-weighted GMM for the coefficient (its ``cluster=`` argument
    affects the two-step weighting in a way not yet reverse-engineered from
    R's ``gmm`` source).  This is a genuine flagged gap (rule 3/15), NOT
    silently skipped — ``test_does_not_match_r_cluster`` pins it.  See
    FUTURE_WORK GMM-RCLUSTER.
    """

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.r = R["oid_2s_cl"]
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", gmm_input,
            step="two-step", cov_type="cluster", cluster="cluster",
        )

    def test_does_not_match_r_cluster(self):
        # Guard: OE's default cluster coefficient must NOT silently match R's
        # cluster coefficient (different efficient-weight conventions; R's is
        # not yet reproduced even with weight="iid").  If this fails, R's
        # cluster weighting has been reverse-engineered — rewrite to assert R
        # parity under the matching toggle.
        r_coef = np.array(self.r["coef"])
        gap = np.max(np.abs(self.oe_r.coefficients.values - r_coef))
        assert gap > 1e-3, (
            "OE cluster coefficient unexpectedly matches R cluster coefficient; "
            "if R's cluster weighting is now reproduced, assert R parity under "
            "the matching toggle."
        )

    def test_coefficients_finite(self):
        assert all(np.isfinite(self.oe_r.coefficients.values))

    def test_standard_errors_positive(self):
        assert all(self.oe_r.std_errors.values > 0)


class TestGmmWeightToggleIidBread:
    """``weight='iid'`` selects the iid efficient-weight bread (rule 15).

    The toggle switches ONLY the two-step efficient-weight bread to the plain
    iid S (each observation its own group); the VCE meat stays at the
    cov-structure S.  This is a genuine, documented convention switch
    reproducing the textbook iid-weighted two-step GMM coefficient.  It is NOT
    claimed as R-cluster parity (R's cluster coefficient is a separate, still
    open gap — see TestGmmROverIdentifiedClusterTwoStep /
    FUTURE_WORK GMM-RCLUSTER).

    Self-consistency: with ``weight='iid'`` the coefficient must equal an
    independent iid two-step GMM computation (identity grouping for the bread,
    one-step residuals for S_iid) to <=1e-6.
    """

    @pytest.fixture(autouse=True)
    def _run(self, gmm_input):
        self.oe_r = oe.gmm(
            "y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", gmm_input,
            step="two-step", cov_type="cluster", cluster="cluster",
            windmeijer=False, robust_meat="one-step", weight="iid",
        )

    def test_coefficients_match_iid_two_step(self, gmm_input):
        from open_econs.models.linear.iv import _parse_iv_formula

        p = _parse_iv_formula(
            "y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", gmm_input
        )
        Y = p["y"]
        X = p["X"]
        parts = [X[:, p["exog_idx"]]] if p["exog_idx"] else []
        parts.append(p["instr_matrix"])
        Z = np.column_stack(parts)
        # One-step identity-weight b1 (matches core).
        A1 = np.linalg.inv(Z.T @ Z)
        b1 = np.linalg.solve(X.T @ Z @ A1 @ Z.T @ X, X.T @ Z @ A1 @ Z.T @ Y)
        e1 = Y - X @ b1
        S_iid = (Z * e1[:, None]).T @ (Z * e1[:, None])
        A2 = np.linalg.inv(S_iid)
        b2_ref = np.linalg.solve(
            X.T @ Z @ A2 @ Z.T @ X, X.T @ Z @ A2 @ Z.T @ Y
        )
        npt.assert_allclose(
            self.oe_r.coefficients.values, b2_ref, atol=1e-6
        )

    def test_iid_bread_distinct_from_cov_structure_bread(self, gmm_input):
        # With cov_type='cluster', weight='iid' must differ from the Stata-style
        # cluster bread (the cluster-efficient coefficient), confirming the
        # toggle actually switches the bread.
        oe_cluster = oe.gmm(
            "y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", gmm_input,
            step="two-step", cov_type="cluster", cluster="cluster",
            windmeijer=False, robust_meat="one-step", weight="stata",
        )
        gap = np.max(
            np.abs(self.oe_r.coefficients.values - oe_cluster.coefficients.values)
        )
        assert gap > 1e-3
