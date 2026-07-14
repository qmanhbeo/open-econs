"""R parity tests for Sun & Abraham (2021) Interaction-Weighted DID.

Validates OE's ``did_sa()`` against R ``fixest::sunab()`` (v0.14.2)
on a staggered DiD panel with 3 treated cohorts and never-treated entities.

Fixture input: ``did_sa_input.csv`` (150 rows, 30 entities, 5 periods;
entities 0-4,20-29 never-treated; 5-9 cohort=2; 10-14 cohort=3; 15-19 cohort=4).

R parity anchor: fixest::sunab() v0.14.2.  No Stata anchor exists for D13.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe
from open_econs.models.linear.fe import _count_absorbed_dof

from ..r_runner import read_r

pytestmark = pytest.mark.r

RTOL = 1e-6

# Load R ground truth once per module (cached).
R_SUNABRAHAM = read_r("did_sa")

INPUT_CSV = (
    Path(__file__).resolve().parents[2]
    / "r" / "fixtures" / "inputs" / "did_sa_input.csv"
)


def _oe_result() -> oe.SaDiDResult:
    """Run did_sa() on the shared staggered-DiD panel."""
    df = pd.read_csv(INPUT_CSV)
    return oe.did_sa(
        data=df,
        y="y",
        cohort="cohort",
        period="time",
        ref_period=-1,
        entity="entity",
        time="time",
        cluster="entity",
        covariates=["x"],
    )


class TestSaDiDBasic:
    """R parity: Sun-Abraham ATT and SE."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()
        self.r = R_SUNABRAHAM

    def test_nobs(self):
        # OE nobs = total data rows (150); R N = estimation sample (75).
        # The estimation sample is the treated-only subset after dropping
        # never-treated (NA cohort) observations.
        assert self.oe.nobs == 150
        assert self.r["N"] == 75

    def test_att(self):
        npt.assert_allclose(self.oe.att, self.r["att"], rtol=RTOL)

    def test_se(self):
        npt.assert_allclose(self.oe.att_se, self.r["se"], rtol=RTOL)

    def test_t_stat(self):
        npt.assert_allclose(self.oe.att_t_stat, self.r["t_stat"], rtol=RTOL)

    def test_p_value(self):
        npt.assert_allclose(self.oe.att_p_value, self.r["p_value"], rtol=RTOL)

    def test_r_squared(self):
        npt.assert_allclose(self.oe.r_squared, self.r["r_squared"], rtol=RTOL)

    def test_sigma2(self):
        npt.assert_allclose(self.oe.sigma2, self.r["sigma2"], rtol=RTOL)


class TestSaDiDCoefficients:
    """R parity: Sun-Abraham raw coefficient vector (9 elements)."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()
        self.r = R_SUNABRAHAM

    def test_raw_coef_names(self):
        assert list(self.oe.coefficients.index) == self.r["raw_coef_names"]

    def test_raw_coefficients(self):
        npt.assert_allclose(
            self.oe.coefficients.values,
            np.array(self.r["raw_coefficients"]),
            rtol=RTOL,
        )


class TestSaDiDVCE:
    """R parity: Sun-Abraham clustered VCE (9x9 matrix)."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()
        self.r = R_SUNABRAHAM
        self.V_oe = self.oe.vcov().values
        self.V_r = np.array(self.r["vce_clustered"]).reshape(
            self.r["vce_nrow"], self.r["vce_ncol"]
        )

    def test_vce_shape(self):
        assert self.V_oe.shape == (9, 9)

    def test_vce_diagonal(self):
        npt.assert_allclose(np.diag(self.V_oe), np.diag(self.V_r), rtol=RTOL)

    def test_vce_offdiagonal(self):
        npt.assert_allclose(self.V_oe, self.V_r, rtol=RTOL)


class TestSaDiDCollinearVars:
    """R parity: collinear variable detection."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()
        self.r = R_SUNABRAHAM

    def test_collinear_variable_names(self):
        """OE's collinear detection matches R's collin.var."""
        r_collin = sorted(self.r["collin_vars"])
        # OE's kept names are the 9 non-collinear coefficients (the result's
        # coefficient index).  The full model has 12 dummy_names + x = 13.
        # Collinear names = full set minus kept.
        # We reconstruct full set from the dummy_names fixture (12 dummies).
        oe_kept = set(self.oe.coefficients.index)
        # Rebuild full set: x + all 12 dummy names (from fixture's raw_coef_names
        # + collin_vars = full dummy set).
        all_dummy_names = self.r["collin_vars"] + [
            n for n in self.r["raw_coef_names"] if n != "x"
        ]
        full_set = set(["x"]) | set(all_dummy_names)
        oe_collin = sorted(full_set - oe_kept)
        assert oe_collin == r_collin


class TestSaDiDDof:
    """Regression guard: nparams, absorbed DOF, residual df."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()
        self.r = R_SUNABRAHAM

    def test_nparams(self):
        """nparams = 9 estimated + 19 absorbed = 28, matching R."""
        k = self.oe.coefficients.shape[0]
        df = pd.read_csv(INPUT_CSV)
        keep_mask = df["cohort"].notna().values
        n_absorbed = _count_absorbed_dof(
            df.loc[df.index[keep_mask]], ["entity", "time"]
        )
        assert k + n_absorbed == self.r["nparams"] == 28

    def test_residual_df(self):
        """Residual df = N_est - nparams = 75 - 28 = 47."""
        k = self.oe.coefficients.shape[0]
        df = pd.read_csv(INPUT_CSV)
        keep_mask = df["cohort"].notna().values
        n_absorbed = _count_absorbed_dof(
            df.loc[df.index[keep_mask]], ["entity", "time"]
        )
        n_est = int(keep_mask.sum())
        assert n_est - (k + n_absorbed) == 47


class TestSaDiDAggregates:
    """R parity: period-level and cohort-level aggregated views."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()
        self.r = R_SUNABRAHAM

    def test_period_coef_names(self):
        assert self.oe.period_names == self.r["period_names"]

    def test_period_coefs(self):
        npt.assert_allclose(
            self.oe.period_coefs,
            np.array(self.r["period_coefs"]),
            rtol=RTOL,
        )

    def test_period_ses(self):
        npt.assert_allclose(
            self.oe.period_ses,
            np.array(self.r["period_ses"]),
            rtol=RTOL,
        )

    def test_cohort_coef_names(self):
        assert self.oe.cohort_names == self.r["cohort_names"]

    def test_cohort_coefs(self):
        npt.assert_allclose(
            self.oe.cohort_coefs,
            np.array(self.r["cohort_coefs"]),
            rtol=RTOL,
        )

    def test_cohort_ses(self):
        npt.assert_allclose(
            self.oe.cohort_ses,
            np.array(self.r["cohort_ses"]),
            rtol=RTOL,
        )


class TestSaDiDSummary:
    """Smoke test: summary() and tidy() return without error."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()

    def test_tidy_shape(self):
        tidy = self.oe.tidy()
        assert tidy.shape == (9, 7)

    def test_summary_string(self):
        s = self.oe.summary()
        assert "Sun" in s
        assert "ATT" in s
