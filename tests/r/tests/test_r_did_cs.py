"""R parity tests for Staggered DiD (R: did::att_gt + did::aggte).

Cell-by-cell ATT(g,t) and SE parity, plus aggregated ATT/SE parity, read from
R-generated JSON fixtures via ``r_runner.read_r``.  Every expected value is
loaded once at module level (cached).

D9 R parity anchor: validates OE's CS2021 DR-DiD implementation against the
estimator's own reference implementation (R ``did`` package v2.5.1,
Callaway & Sant'Anna 2021).  The three-way comparison (OE / Stata ``csdid`` /
R ``did``) is documented in the did() kickoff brief.

Source: R ``did`` package v2.5.1, ``att_gt()`` with ``est_method="dr"``,
``control_group="nevertreated"``, ``base_period="varying"``.
Aggregation: ``aggte(type="simple")`` for the overall ATT/SE.
"""

from __future__ import annotations

from pathlib import Path

import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..r_runner import read_r

pytestmark = pytest.mark.r

RTOL = 1e-6

# Load R ground truth once per module (cached).
R_BAL = read_r("did_cs_balanced")
R_UNBAL = read_r("did_cs_unbalanced")


def _oe_att_gt(oe_r: oe.CsDiDResult) -> dict[tuple[int, int], float]:
    gt = oe_r.att_group_time
    return {(int(r.cohort), int(r.time)): r.att for r in gt.itertuples()}


def _oe_se_gt(oe_r: oe.CsDiDResult) -> dict[tuple[int, int], float]:
    gt = oe_r.att_group_time
    return {(int(r.cohort), int(r.time)): r.se for r in gt.itertuples()}


class TestCsDiDRParityBalanced:
    """R parity: doubly-robust (dripw) with covariates x, z — balanced panel.

    Validates against R ``did::att_gt()`` + ``aggte(type="simple")`` on the
    same balanced fixture data used by the Stata ``csdid`` tests (20 entities,
    g=3 treated at t=3, g=5 excluded).
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        df = pd.read_csv(
            Path(__file__).resolve().parents[2] / "stata" / "fixtures" / "inputs" / "df_panel.csv"
        )
        # Balanced: keep entities 0-19 (gvar=5 entities 20-29 excluded)
        df = df[df["entity"] < 20].copy()
        df["treat"] = 0.0
        df.loc[(df["entity"] >= 10) & (df["time"] >= 3), "treat"] = 1.0
        self.oe_r = oe.did_cs(
            df, y="y", entity="entity", time="time", treatment="treat",
            covariates=["x", "z"], method="dripw",
            control_cohorts="never_treated",
        )

    def test_cell_att_g3_t3(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], R_BAL["b_g3_t2_3"], rtol=RTOL)

    def test_cell_att_g3_t4(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], R_BAL["b_g3_t3_4"], rtol=RTOL)

    def test_cell_se_g3_t3(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], R_BAL["se_g3_t2_3"], rtol=RTOL)

    def test_cell_se_g3_t4(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], R_BAL["se_g3_t3_4"], rtol=RTOL)

    def test_aggregated_att_simple(self):
        npt.assert_allclose(self.oe_r.att, R_BAL["agg_att_simple"], rtol=RTOL)

    def test_aggregated_se(self):
        """D9 highest-value assertion: OE IF-based SE vs R did::aggte overall.se."""
        npt.assert_allclose(self.oe_r.att_se, R_BAL["agg_se_simple"], rtol=RTOL)

    def test_cell_count(self):
        assert len(self.oe_r.att_group_time) == 2


class TestCsDiDRParityUnbalanced:
    """R parity: doubly-robust (dripw) with covariates — unbalanced panel.

    Validates against R ``did::att_gt(allow_unbalanced_panel=TRUE)`` on the
    unbalanced fixture (23 entities, 15 never-treated + 8 g=3).
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        df = pd.read_csv(
            Path(__file__).resolve().parents[2] / "stata" / "fixtures" / "inputs" / "df_panel_unbalanced.csv"
        )
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
        npt.assert_allclose(gt[(3, 3)], R_UNBAL["b_g3_t2_3"], rtol=RTOL)

    def test_unbalanced_cell_att_g3_t4(self):
        gt = _oe_att_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], R_UNBAL["b_g3_t3_4"], rtol=RTOL)

    def test_unbalanced_cell_se_g3_t3(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 3)], R_UNBAL["se_g3_t2_3"], rtol=RTOL)

    def test_unbalanced_cell_se_g3_t4(self):
        gt = _oe_se_gt(self.oe_r)
        npt.assert_allclose(gt[(3, 4)], R_UNBAL["se_g3_t3_4"], rtol=RTOL)

    def test_unbalanced_aggregated_att(self):
        npt.assert_allclose(self.oe_r.att, R_UNBAL["agg_att_simple"], rtol=RTOL)

    def test_unbalanced_aggregated_se(self):
        npt.assert_allclose(self.oe_r.att_se, R_UNBAL["agg_se_simple"], rtol=RTOL)

    def test_unbalanced_cell_count(self):
        assert len(self.oe_r.att_group_time) == 2
