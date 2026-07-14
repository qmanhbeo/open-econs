"""R parity tests for Gardner (2022) DID2S.

Validates OE's ``did_gardner()`` against R ``did2s::did2s()`` (v1.2.1) on
the same balanced panel used by the staggered-DiD tests.  Entities 0-9 are
never-treated; entities 10-19 are treated at time >= 3.

The Gardner DID2S estimator uses a two-stage IF for cluster-robust SEs:
    IF = IF_fs - IF_ss
where IF_fs accounts for first-stage estimation uncertainty.  A naive
single-stage cluster-robust VCE (IF_ss only) will underestimate the SE.
Source-confirmed against R ``did2s:::did2s()``.

No Stata anchor exists for Gardner DID2S.  The sole parity anchor is R
``did2s`` package v1.2.1.

References
----------
Gardner, John. 2022. "Two-Stage Differences in Differences."
arXiv:2207.05943. Working paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..r_runner import read_r

pytestmark = pytest.mark.r

RTOL = 1e-6

# Load R ground truth once per module (cached).
R_GARDNER = read_r("did_gardner")

INPUT_CSV = (
    Path(__file__).resolve().parents[2]
    / "r" / "fixtures" / "inputs" / "did_gardner_input.csv"
)


def _oe_result() -> oe.GardnerResult:
    """Run did_gardner() on the shared balanced panel."""
    df = pd.read_csv(INPUT_CSV)
    df["treat"] = (
        (df["entity"] >= 10) & (df["entity"] < 20) & (df["time"] >= 3)
    ).astype(int)
    return oe.did_gardner(
        data=df,
        y="y",
        first_stage="0 + C(entity) + C(time)",
        second_stage="treat",
        treatment="treat",
        cluster="entity",
    )


class TestGardnerBasic:
    """R parity: Gardner DID2S basic ATT and SE."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()
        self.r = R_GARDNER

    def test_nobs(self):
        assert self.oe.nobs == self.r["N"]

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

    def test_adj_r_squared(self):
        npt.assert_allclose(
            self.oe.adj_r_squared, self.r["adj_r_squared"], rtol=RTOL,
        )

    def test_sigma2(self):
        npt.assert_allclose(self.oe.sigma2, self.r["sigma2"], rtol=RTOL)


class TestGardnerCoefficients:
    """R parity: Gardner DID2S coefficient table."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()
        self.r = R_GARDNER

    def test_coef_names(self):
        assert list(self.oe.coefficients.index) == [self.r["coef_names"]]

    def test_coef_value(self):
        npt.assert_allclose(
            self.oe.coefficients.values, self.r["att"], rtol=RTOL,
        )

    def test_se_value(self):
        npt.assert_allclose(
            self.oe.std_errors.values, self.r["se"], rtol=RTOL,
        )


class TestGardnerVCE:
    """R parity: Gardner DID2S variance-covariance matrices."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()
        self.r = R_GARDNER

    def test_vce_clustered(self):
        """Cluster-robust VCE matches R's cov.scaled.

        This is the critical test: the two-stage IF must include IF_fs
        (first-stage estimation uncertainty).  A naive single-stage
        VCE would give ~0.1757 instead of the correct 0.2526.
        """
        npt.assert_allclose(
            self.oe.vcov().values[0, 0],
            self.r["vce_clustered"][0],
            rtol=RTOL,
        )


class TestGardnerSummary:
    """Smoke test: summary() and tidy() return without error."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result()

    def test_tidy_shape(self):
        tidy = self.oe.tidy()
        assert tidy.shape == (1, 7)

    def test_summary_string(self):
        s = self.oe.summary()
        assert "Gardner" in s
        assert "ATT" in s
