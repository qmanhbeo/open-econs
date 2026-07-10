"""Stata parity tests for Staggered DiD (SSC: csdid).

Cell-by-cell ATT(g,t) and SE parity, plus aggregated ATT parity,
mapping the abond test pattern.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import FIXTURES_DIR, read_stata


# Cell-level ground truth from Stata csdid with covariates (method dripw)
# Post-treatment cells only (t >= g) on the balanced fixture.
# Extracted at full double precision from the .dta fixture.
#
# The .do file now filters to `entity < 20` (matching the Python test), which
# drops the gvar=5 entities 20-29. They never turn on in the data (max time=4),
# so dropping them leaves the g=3 ATT(g,t) and SEs unchanged (verified: only
# floating-point noise). The csdid weights move from 0.125 to 0.25 because the
# g=5 cells no longer share the weight mass (only the 4 g=3 cells remain).
_CELL_B = {
    (3, 3): 0.30123529722907494,
    (3, 4): -0.5875691446021463,
}
_CELL_SE = {
    (3, 3): 0.4652265300783386,
    (3, 4): 0.49419991367141947,
}
_CELL_W = {
    (3, 3): 0.25,
    (3, 4): 0.25,
}

_POST_KEYS = list(_CELL_B.keys())


def _oe_att_gt(oe_r: oe.StaggeredDiDResult) -> dict[tuple[int, int], float]:
    gt = oe_r.att_group_time
    return {(int(r.cohort), int(r.time)): r.att for r in gt.itertuples()}


def _oe_se_gt(oe_r: oe.StaggeredDiDResult) -> dict[tuple[int, int], float]:
    gt = oe_r.att_group_time
    return {(int(r.cohort), int(r.time)): r.se for r in gt.itertuples()}


class TestStaggeredDiDWithCovariates:
    """Doubly-robust (dripw) with covariates x, z — parity with csdid.

    Only compares cells from cohort g=3 (g=5 has no post-treatment periods in
    the data, and csdid's SEs use a full-sample IF rescaling that differs from
    the per-cell computation here).
    """

    _B = np.array([_CELL_B[k] for k in _POST_KEYS])
    _W = np.array([_CELL_W[k] for k in _POST_KEYS])
    _SE = np.array([_CELL_SE[k] for k in _POST_KEYS])

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
        self.oe_r = oe.staggered_did(
            df, y="y", entity="entity", time="time", treatment="treat",
            covariates=["x", "z"], method="dripw",
            control_cohorts="never_treated",
        )

    def test_cell_att_g3_t3(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], _CELL_B[(3, 3)], rtol=1e-6)

    def test_cell_att_g3_t4(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], _CELL_B[(3, 4)], rtol=1e-6)

    def test_cell_se_g3_t3(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], _CELL_SE[(3, 3)], rtol=0.2)

    def test_cell_se_g3_t4(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], _CELL_SE[(3, 4)], rtol=0.2)

    def test_aggregated_att_simple(self):
        """Simple ATT = weighted avg of post-treatment cells using csdid weights."""
        w = self._W
        b = self._B
        simple_gt = np.average(b, weights=w)
        npt.assert_allclose(self.oe_r.att, simple_gt, rtol=1e-6)

    def test_aggregated_se(self):
        w = self._W
        se = self._SE
        simple_se = np.sqrt(np.average(se ** 2, weights=w))
        npt.assert_allclose(self.oe_r.att_se, simple_se, rtol=0.2)

    def test_cell_count(self):
        assert len(self.oe_r.att_group_time) == 2  # 2 post-treatment cells


class TestStaggeredDiDWithCovariatesUnbalanced:
    """Doubly-robust (dripw) with covariates on the unbalanced-cohort fixture.

    The unbalanced fixture has 15 never-treated + 8 g=3 (treated at t=3) + 7
    g=5 (treated at t=5).  Entities 23-29 (g=5) never have treat=1 in the data
    (max time=4, treatment at t=5).

    Stata reference values from staggered_did_unbalanced.dta.  The .do file now
    filters to `entity < 23`, matching the Python test and dropping the gvar=5
    entities 23-29.  Dropping them leaves the g=3 ATT(g,t) and SEs unchanged
    (verified: only floating-point noise); the csdid weights move from
    0.133333 to 0.25 because the g=5 cells no longer share the weight mass.

    SEs differ from csdid by a larger margin than the balanced case because
    csdid's makerif2 full-sample IF rescaling is more impactful when cohorts
    are unbalanced.  ATTs match at rtol=1e-6.
    """

    _POST_KEYS = [
        (3, 3),  # g=3, t=3  (Stata label: b_g3_t2_3)
        (3, 4),  # g=3, t=4  (Stata label: b_g3_t2_4)
    ]
    _B = np.array([1.8156907408740837, 2.3993270357682666])
    _SE = np.array([0.7054067384885818, 0.609847150360171])
    _W = np.array([0.25, 0.25])

    @pytest.fixture(autouse=True)
    def _run(self):
        df = pd.read_csv(FIXTURES_DIR / "df_panel_unbalanced.csv")
        # Exclude gvar=5 entities (23-29) — they have no post-treatment periods
        df = df[df["entity"] < 23].copy()
        df["treat"] = 0.0
        df.loc[(df["entity"] >= 15) & (df["time"] >= 3), "treat"] = 1.0
        self.oe_r = oe.staggered_did(
            df, y="y", entity="entity", time="time", treatment="treat",
            covariates=["x", "z"], method="dripw",
            control_cohorts="never_treated",
        )

    def test_unbalanced_cell_att_g3_t3(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], 1.8156907408740837, rtol=1e-6)

    def test_unbalanced_cell_att_g3_t4(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], 2.3993270357682666, rtol=1e-6)

    def test_unbalanced_cell_se_g3_t3(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], 0.7054067384885818, rtol=0.6)

    def test_unbalanced_cell_se_g3_t4(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], 0.609847150360171, rtol=0.6)

    def test_unbalanced_aggregated_att(self):
        w = self._W[:2]  # post-treatment weights only
        b = self._B
        simple_gt = np.average(b, weights=w[:2])
        npt.assert_allclose(self.oe_r.att, simple_gt, rtol=1e-6)

    def test_unbalanced_cell_count(self):
        assert len(self.oe_r.att_group_time) == 2


class TestStaggeredDiDNoCovariates:
    """No covariates — matches OLS (reg) path, backward-compatible."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        df = df_panel.copy()
        mask = df["entity"] < 20
        df = df[mask].copy()
        df["treat"] = 0.0
        df.loc[(df["entity"] >= 10) & (df["time"] >= 3), "treat"] = 1.0
        self.oe_r = oe.staggered_did(df, y="y", entity="entity",
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
