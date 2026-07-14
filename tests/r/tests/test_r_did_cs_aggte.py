"""R parity tests for Staggered DiD aggte() — dynamic/group/calendar aggregation.

Validates OE's ``StaggeredDiDResult.aggte()`` against R ``did::aggte()`` on the
same fixture data used by the existing staggered-DiD R parity tests.  Three
aggregation types are tested: ``dynamic`` (event-time), ``group`` (cohort), and
``calendar`` (time period).

No Stata anchor exists for ``aggte()`` — Stata's ``csdid`` does not implement
dynamic/group/calendar aggregation.  The sole parity anchor is R ``did``
package v2.5.1, ``aggte()`` with ``est_method="dr"``.

References
----------
Callaway, Brantly and Pedro H.C. Sant'Anna. 2021.
"Difference-in-Differences with Multiple Time Periods."
*Journal of Econometrics*, Vol. 225, No. 2, pp. 200-230.
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
R_BAL = read_r("did_cs_balanced")
R_UNBAL = read_r("did_cs_unbalanced")

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "stata" / "fixtures" / "inputs"


def _oe_result_balanced() -> oe.CsDiDResult:
    df = pd.read_csv(FIXTURES_DIR / "df_panel.csv")
    df = df[df["entity"] < 20].copy()
    df["treat"] = 0.0
    df.loc[(df["entity"] >= 10) & (df["time"] >= 3), "treat"] = 1.0
    return oe.did_cs(
        df, y="y", entity="entity", time="time", treatment="treat",
        covariates=["x", "z"], method="dripw",
        control_cohorts="never_treated",
    )


def _oe_result_unbalanced() -> oe.CsDiDResult:
    df = pd.read_csv(FIXTURES_DIR / "df_panel_unbalanced.csv")
    df = df[df["entity"] < 23].copy()
    df["treat"] = 0.0
    df.loc[(df["entity"] >= 15) & (df["time"] >= 3), "treat"] = 1.0
    return oe.did_cs(
        df, y="y", entity="entity", time="time", treatment="treat",
        covariates=["x", "z"], method="dripw",
        control_cohorts="never_treated",
    )


class TestAggteDynamicBalanced:
    """R parity: dynamic (event-time) aggregation — balanced panel.

    Validates against R ``did::aggte(type="dynamic")`` on the balanced fixture.
    Only post-treatment event times (e >= 0) are validated; OE does not compute
    pre-treatment cells that R includes in its dynamic output.
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result_balanced().aggte(type="dynamic")
        self.r = R_BAL

    def test_type(self):
        assert self.oe.type == "dynamic"

    def test_overall_att(self):
        npt.assert_allclose(self.oe.att, self.r["agg_dynamic_overall_att"], rtol=RTOL)

    def test_overall_se(self):
        npt.assert_allclose(self.oe.se, self.r["agg_dynamic_overall_se"], rtol=RTOL)

    def test_att_e0(self):
        att_by = self.oe.att_by.set_index("lead")
        npt.assert_allclose(att_by.loc[0, "att"], self.r["agg_dynamic_att_e0"], rtol=RTOL)

    def test_att_e1(self):
        att_by = self.oe.att_by.set_index("lead")
        npt.assert_allclose(att_by.loc[1, "att"], self.r["agg_dynamic_att_e1"], rtol=RTOL)

    def test_se_e0(self):
        att_by = self.oe.att_by.set_index("lead")
        npt.assert_allclose(att_by.loc[0, "se"], self.r["agg_dynamic_se_e0"], rtol=RTOL)

    def test_se_e1(self):
        att_by = self.oe.att_by.set_index("lead")
        npt.assert_allclose(att_by.loc[1, "se"], self.r["agg_dynamic_se_e1"], rtol=RTOL)

    def test_level_count(self):
        assert len(self.oe.att_by) == 2


class TestAggteGroupBalanced:
    """R parity: group (cohort) aggregation — balanced panel.

    Validates against R ``did::aggte(type="group")`` on the balanced fixture.
    Single treated cohort (g=3), so group ATT equals the overall ATT.
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result_balanced().aggte(type="group")
        self.r = R_BAL

    def test_type(self):
        assert self.oe.type == "group"

    def test_overall_att(self):
        npt.assert_allclose(self.oe.att, self.r["agg_group_overall_att"], rtol=RTOL)

    def test_overall_se(self):
        npt.assert_allclose(self.oe.se, self.r["agg_group_overall_se"], rtol=RTOL)

    def test_att_g3(self):
        att_by = self.oe.att_by.set_index("cohort")
        npt.assert_allclose(att_by.loc[3, "att"], self.r["agg_group_att_g3"], rtol=RTOL)

    def test_se_g3(self):
        att_by = self.oe.att_by.set_index("cohort")
        npt.assert_allclose(att_by.loc[3, "se"], self.r["agg_group_se_g3"], rtol=RTOL)

    def test_level_count(self):
        assert len(self.oe.att_by) == 1


class TestAggteCalendarBalanced:
    """R parity: calendar (time) aggregation — balanced panel.

    Validates against R ``did::aggte(type="calendar")`` on the balanced fixture.
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result_balanced().aggte(type="calendar")
        self.r = R_BAL

    def test_type(self):
        assert self.oe.type == "calendar"

    def test_overall_att(self):
        npt.assert_allclose(self.oe.att, self.r["agg_calendar_overall_att"], rtol=RTOL)

    def test_overall_se(self):
        npt.assert_allclose(self.oe.se, self.r["agg_calendar_overall_se"], rtol=RTOL)

    def test_att_t3(self):
        att_by = self.oe.att_by.set_index("time")
        npt.assert_allclose(att_by.loc[3, "att"], self.r["agg_calendar_att_t3"], rtol=RTOL)

    def test_att_t4(self):
        att_by = self.oe.att_by.set_index("time")
        npt.assert_allclose(att_by.loc[4, "att"], self.r["agg_calendar_att_t4"], rtol=RTOL)

    def test_se_t3(self):
        att_by = self.oe.att_by.set_index("time")
        npt.assert_allclose(att_by.loc[3, "se"], self.r["agg_calendar_se_t3"], rtol=RTOL)

    def test_se_t4(self):
        att_by = self.oe.att_by.set_index("time")
        npt.assert_allclose(att_by.loc[4, "se"], self.r["agg_calendar_se_t4"], rtol=RTOL)

    def test_level_count(self):
        assert len(self.oe.att_by) == 2


class TestAggteDynamicUnbalanced:
    """R parity: dynamic (event-time) aggregation — unbalanced panel.

    Validates against R ``did::aggte(type="dynamic")`` on the unbalanced fixture.
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result_unbalanced().aggte(type="dynamic")
        self.r = R_UNBAL

    def test_type(self):
        assert self.oe.type == "dynamic"

    def test_overall_att(self):
        npt.assert_allclose(self.oe.att, self.r["agg_dynamic_overall_att"], rtol=RTOL)

    def test_overall_se(self):
        npt.assert_allclose(self.oe.se, self.r["agg_dynamic_overall_se"], rtol=RTOL)

    def test_att_e0(self):
        att_by = self.oe.att_by.set_index("lead")
        npt.assert_allclose(att_by.loc[0, "att"], self.r["agg_dynamic_att_e0"], rtol=RTOL)

    def test_att_e1(self):
        att_by = self.oe.att_by.set_index("lead")
        npt.assert_allclose(att_by.loc[1, "att"], self.r["agg_dynamic_att_e1"], rtol=RTOL)

    def test_se_e0(self):
        att_by = self.oe.att_by.set_index("lead")
        npt.assert_allclose(att_by.loc[0, "se"], self.r["agg_dynamic_se_e0"], rtol=RTOL)

    def test_se_e1(self):
        att_by = self.oe.att_by.set_index("lead")
        npt.assert_allclose(att_by.loc[1, "se"], self.r["agg_dynamic_se_e1"], rtol=RTOL)

    def test_level_count(self):
        assert len(self.oe.att_by) == 2


class TestAggteGroupUnbalanced:
    """R parity: group (cohort) aggregation — unbalanced panel.

    Validates against R ``did::aggte(type="group")`` on the unbalanced fixture.
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result_unbalanced().aggte(type="group")
        self.r = R_UNBAL

    def test_type(self):
        assert self.oe.type == "group"

    def test_overall_att(self):
        npt.assert_allclose(self.oe.att, self.r["agg_group_overall_att"], rtol=RTOL)

    def test_overall_se(self):
        npt.assert_allclose(self.oe.se, self.r["agg_group_overall_se"], rtol=RTOL)

    def test_att_g3(self):
        att_by = self.oe.att_by.set_index("cohort")
        npt.assert_allclose(att_by.loc[3, "att"], self.r["agg_group_att_g3"], rtol=RTOL)

    def test_se_g3(self):
        att_by = self.oe.att_by.set_index("cohort")
        npt.assert_allclose(att_by.loc[3, "se"], self.r["agg_group_se_g3"], rtol=RTOL)

    def test_level_count(self):
        assert len(self.oe.att_by) == 1


class TestAggteCalendarUnbalanced:
    """R parity: calendar (time) aggregation — unbalanced panel.

    Validates against R ``did::aggte(type="calendar")`` on the unbalanced fixture.
    """

    @pytest.fixture(autouse=True)
    def _run(self):
        self.oe = _oe_result_unbalanced().aggte(type="calendar")
        self.r = R_UNBAL

    def test_type(self):
        assert self.oe.type == "calendar"

    def test_overall_att(self):
        npt.assert_allclose(self.oe.att, self.r["agg_calendar_overall_att"], rtol=RTOL)

    def test_overall_se(self):
        npt.assert_allclose(self.oe.se, self.r["agg_calendar_overall_se"], rtol=RTOL)

    def test_att_t3(self):
        att_by = self.oe.att_by.set_index("time")
        npt.assert_allclose(att_by.loc[3, "att"], self.r["agg_calendar_att_t3"], rtol=RTOL)

    def test_att_t4(self):
        att_by = self.oe.att_by.set_index("time")
        npt.assert_allclose(att_by.loc[4, "att"], self.r["agg_calendar_att_t4"], rtol=RTOL)

    def test_se_t3(self):
        att_by = self.oe.att_by.set_index("time")
        npt.assert_allclose(att_by.loc[3, "se"], self.r["agg_calendar_se_t3"], rtol=RTOL)

    def test_se_t4(self):
        att_by = self.oe.att_by.set_index("time")
        npt.assert_allclose(att_by.loc[4, "se"], self.r["agg_calendar_se_t4"], rtol=RTOL)

    def test_level_count(self):
        assert len(self.oe.att_by) == 2


class TestAggteInvalidType:
    """Validation: invalid aggregation type raises ValueError."""

    def test_invalid_type(self):
        df = pd.read_csv(FIXTURES_DIR / "df_panel.csv")
        df = df[df["entity"] < 20].copy()
        df["treat"] = 0.0
        df.loc[(df["entity"] >= 10) & (df["time"] >= 3), "treat"] = 1.0
        result = oe.did_cs(
            df, y="y", entity="entity", time="time", treatment="treat",
            covariates=["x", "z"], method="dripw",
            control_cohorts="never_treated",
        )
        with pytest.raises(ValueError, match="type must be one of"):
            result.aggte(type="invalid")
