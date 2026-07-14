"""Stata parity tests for Staggered DiD (SSC: csdid).

Cell-by-cell ATT(g,t) and SE parity, plus aggregated ATT/SE parity, read live
from Stata-generated .dta fixtures via ``read_stata``.  Every expected value is
loaded once at module level (cached) from ``did_cs.dta`` /
``did_cs_unbalanced.dta``; the aggregated SE is recomputed in Stata from
csdid's per-entity RIF using open_econs' exact aggregation formula.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..stata_runner import INPUTS_DIR, read_stata

pytestmark = pytest.mark.stata

RTOL = 1e-6

# Load Stata ground truth once per module (cached), not per test method.
S = read_stata("did_cs")
S_U = read_stata("did_cs_unbalanced")


def _oe_att_gt(oe_r: oe.CsDiDResult) -> dict[tuple[int, int], float]:
    gt = oe_r.att_group_time
    return {(int(r.cohort), int(r.time)): r.att for r in gt.itertuples()}


def _oe_se_gt(oe_r: oe.CsDiDResult) -> dict[tuple[int, int], float]:
    gt = oe_r.att_group_time
    return {(int(r.cohort), int(r.time)): r.se for r in gt.itertuples()}


class TestCsDiDWithCovariates:
    """Doubly-robust (dripw) with covariates x, z — parity with csdid.

    Only compares cells from cohort g=3 (g=5 has no post-treatment periods in
    the data, and csdid's SEs use a full-sample IF rescaling that differs from
    the per-cell computation here).
    """

    _POST_KEYS = [(3, 3), (3, 4)]

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        df = df_panel.copy()
        # Only keep entities 0-19 (gvar=0 never-treated + gvar=3 treated at t=3).
        # Entities 20-29 (gvar=5, treated at t=5) are excluded because they never
        # turn on in the data (max time=4), and csdid handles them via explicit
        # gvar, not via the treatment indicator.
        mask = df["entity"] < 20
        df = df[mask].copy()
        df["treat"] = 0.0
        df.loc[(df["entity"] >= 10) & (df["time"] >= 3), "treat"] = 1.0
        self.oe_r = oe.did_cs(
            df, y="y", entity="entity", time="time", treatment="treat",
            covariates=["x", "z"], method="dripw",
            control_cohorts="never_treated",
        )

    def test_cell_att_g3_t3(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], S["b_g3_t2_3"], rtol=RTOL)

    def test_cell_att_g3_t4(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], S["b_g3_t2_4"], rtol=RTOL)

    def test_cell_se_g3_t3(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], S["se_g3_t2_3"], rtol=RTOL)

    def test_cell_se_g3_t4(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], S["se_g3_t2_4"], rtol=RTOL)

    def test_aggregated_att_simple(self):
        """Simple ATT = weighted avg of post-treatment cells using csdid weights."""
        b = np.array([S["b_g3_t2_3"], S["b_g3_t2_4"]])
        w = np.array([S["w_g3_t2_3"], S["w_g3_t2_4"]])
        simple_gt = np.average(b, weights=w)
        npt.assert_allclose(self.oe_r.att, simple_gt, rtol=RTOL)

    def test_aggregated_se(self):
        # Aggregated SE matches csdid's influence-function aggregation, i.e.
        # `csdid y x z, saverif(rif)` + `csdid_stats simple` (and the `did` R
        # package's aggte(type="simple"), getSE = sqrt(mean(if^2)/n)).  This is
        # the value csdid itself reports from its saved RIFs = 0.41781627.
        # NOTE: Stata's `csdid_estat simple` is buggy in csdid v1.6/v1.58 — it
        # prints the first (pre-treatment) cell's SE (0.7479047), NOT an
        # aggregation SE.  Do not compare against `csdid_estat simple`.
        npt.assert_allclose(self.oe_r.att_se, S["agg_se"], rtol=RTOL)

    def test_cell_count(self):
        assert len(self.oe_r.att_group_time) == 2  # 2 post-treatment cells


class TestCsDiDWithCovariatesUnbalanced:
    """Doubly-robust (dripw) with covariates on the unbalanced-cohort fixture.

    The unbalanced fixture has 15 never-treated + 8 g=3 (treated at t=3) + 7
    g=5 (treated at t=5).  Entities 23-29 (g=5) never have treat=1 in the data
    (max time=4, treatment at t=5).

    The aggregated SE matches csdid's `saverif`+`csdid_stats simple` IF
    aggregation (0.62720813); `csdid_estat simple` mis-reports a different
    value (0.47824472) for this fixture.  ATTs match at rtol=1e-6.
    """

    _POST_KEYS = [(3, 3), (3, 4)]

    @pytest.fixture(autouse=True)
    def _run(self):
        df = pd.read_csv(INPUTS_DIR / "df_panel_unbalanced.csv")
        # Exclude gvar=5 entities (23-29) — they have no post-treatment periods
        df = df[df["entity"] < 23].copy()
        df["treat"] = 0.0
        df.loc[(df["entity"] >= 15) & (df["time"] >= 3), "treat"] = 1.0
        self.oe_r = oe.did_cs(
            df, y="y", entity="entity", time="time", treatment="treat",
            covariates=["x", "z"], method="dripw",
            control_cohorts="never_treated",
        )

    def test_unbalanced_cell_att_g3_t3(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], S_U["b_g3_t2_3"], rtol=RTOL)

    def test_unbalanced_cell_att_g3_t4(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], S_U["b_g3_t2_4"], rtol=RTOL)

    def test_unbalanced_cell_se_g3_t3(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], S_U["se_g3_t2_3"], rtol=RTOL)

    def test_unbalanced_cell_se_g3_t4(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], S_U["se_g3_t2_4"], rtol=RTOL)

    def test_unbalanced_aggregated_att(self):
        b = np.array([S_U["b_g3_t2_3"], S_U["b_g3_t2_4"]])
        w = np.array([S_U["w_g3_t2_3"], S_U["w_g3_t2_4"]])
        simple_gt = np.average(b, weights=w)
        npt.assert_allclose(self.oe_r.att, simple_gt, rtol=RTOL)

    def test_unbalanced_aggregated_se(self):
        # Same as the balanced case: matches csdid's `saverif`+`csdid_stats
        # simple` IF aggregation (= 0.62720813), NOT `csdid_estat simple`
        # (which mis-reports 0.47824472 for this fixture).  Extracted live from
        # csdid's per-entity RIF via makerif2/aggte/make_tbl in
        # did_cs_unbalanced.do.
        npt.assert_allclose(self.oe_r.att_se, S_U["agg_se"], rtol=RTOL)

    def test_unbalanced_cell_count(self):
        assert len(self.oe_r.att_group_time) == 2


class TestCsDiDNoCovariates:
    """No covariates — matches OLS (reg) path, backward-compatible."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        df = df_panel.copy()
        mask = df["entity"] < 20
        df = df[mask].copy()
        df["treat"] = 0.0
        df.loc[(df["entity"] >= 10) & (df["time"] >= 3), "treat"] = 1.0
        self.oe_r = oe.did_cs(df, y="y", entity="entity",
                                     time="time", treatment="treat")

    def test_att_not_nan(self):
        assert np.isfinite(self.oe_r.att)

    def test_att_se_not_nan(self):
        assert np.isfinite(self.oe_r.att_se)

    def test_att_positive_se(self):
        assert self.oe_r.att_se > 0

    def test_cell_count(self):
        assert len(self.oe_r.att_group_time) == 2

    def test_att_reasonable(self):
        assert -5 < self.oe_r.att < 5
