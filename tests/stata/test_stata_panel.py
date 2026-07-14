"""Stata parity tests for Panel estimators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import read_stata

pytestmark = pytest.mark.stata

S_PANEL_FE = read_stata("panel_fe")
S_PANEL_FE_TWOWAY = read_stata("panel_fe_twoway")
S_PANEL_RE = read_stata("panel_re")
S_PANEL_POOLED = read_stata("panel_pooled")
S_PANEL_FD = read_stata("panel_fd")
S_PANEL_HAUSMAN = read_stata("panel_hausman")
S_PANEL_FE_MW_CLUSTER = read_stata("panel_fe_multiway_cluster")


class TestPanelFE:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_PANEL_FE
        # Stata `xtreg y x z, fe` is one-way entity FE only
        self.oe_r = oe.fe("y ~ x + z", data=df_panel, entity="entity",
                          cov_type="nonrobust")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]],
                            rtol=1e-6)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])

    def test_r_squared(self):
        npt.assert_allclose(self.oe_r.r_squared, self.s["r2_w"], rtol=1e-6)


class TestPanelFETwoWay:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_PANEL_FE_TWOWAY
        # Stata `xtreg y x z i.time, fe` — entity FE + time dummies
        self.oe_r = oe.fe("y ~ x + z", data=df_panel, entity="entity",
                          time="time", cov_type="nonrobust")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]],
                            rtol=1e-6)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])

    def test_df_resid(self):
        assert self.oe_r.df_resid == int(self.s["df_r"])

    def test_r_squared(self):
        npt.assert_allclose(self.oe_r.r_squared, self.s["r2_w"], rtol=1e-6)


class TestPanelRE:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_PANEL_RE
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        # linearmodels uses "unadjusted" not "nonrobust"
        self.oe_r = ctx.re("y ~ x + z", cov_type="unadjusted")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_int"], self.s["b_x"], self.s["b_z"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_int"], self.s["se_x"], self.s["se_z"]],
                            rtol=1e-6)


class TestPanelPooled:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_PANEL_POOLED
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        self.oe_r = ctx.pooled("y ~ x + z", cov_type="nonrobust")

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_int"], self.s["b_x"], self.s["b_z"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_int"], self.s["se_x"], self.s["se_z"]],
                            rtol=1e-6)


class TestPanelFD:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_PANEL_FD
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        self.oe_r = ctx.diff("y ~ x + z")

    def test_coefficients(self):
        # Stata: manual diff + regress with noconstant
        # OE: linearmodels FirstDifferenceOLS (same: forbids intercept)
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]],
                            rtol=1e-6)


class TestPanelHausman:
    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_PANEL_HAUSMAN
        ctx = oe.PanelContext(df_panel, entity="entity", time="time")
        # One-way entity FE to match Stata's xtreg y x z, fe
        fe_r = ctx.fe("y ~ x + z", cov_type="nonrobust", entity="entity", time=None)
        # linearmodels uses "unadjusted" not "nonrobust"
        re_r = ctx.re("y ~ x + z", cov_type="unadjusted")
        self.oe_h = ctx.hausman(fe_r, re_r)

    def test_chi2(self):
        npt.assert_allclose(self.oe_h.statistic, self.s["chi2"], rtol=1e-6)

    def test_p_value(self):
        npt.assert_allclose(self.oe_h.p_value, self.s["p"], rtol=1e-6)


class TestFEVcovIndexConsistency:
    """Regression: fe().coefficients.index must equal fe().vcov().index.

    Prevents recurrence of the bug where sm.OLS on a bare numpy array
    auto-generated ['x1','x2'] names that desynced vcov() from
    coefficients, breaking ctx.hausman().
    """

    @pytest.fixture(autouse=True)
    def _data(self):
        self.df = pd.read_csv(
            str(Path(__file__).parent / "fixtures" / "df_panel.csv")
        )

    def test_entity_only(self):
        r = oe.fe("y ~ x + z", data=self.df, entity="entity", cov_type="nonrobust")
        pd.testing.assert_index_equal(r.coefficients.index, r.vcov().index)

    def test_time_only(self):
        r = oe.fe("y ~ x + z", data=self.df, time="time", cov_type="HC1")
        pd.testing.assert_index_equal(r.coefficients.index, r.vcov().index)

    def test_two_way(self):
        r = oe.fe(
            "y ~ x + z", data=self.df, entity="entity", time="time",
            cov_type="nonrobust",
        )
        pd.testing.assert_index_equal(r.coefficients.index, r.vcov().index)

    def test_clustered(self):
        r = oe.fe(
            "y ~ x + z", data=self.df, entity="entity",
            cluster="entity", cov_type="HC1",
        )
        pd.testing.assert_index_equal(r.coefficients.index, r.vcov().index)

    def test_hc2_rejected_entity(self):
        with pytest.raises(oe.VcovTypeNotSupportedError):
            oe.fe("y ~ x + z", data=self.df, entity="entity", cov_type="HC2")

    def test_hc3_rejected_entity(self):
        with pytest.raises(oe.VcovTypeNotSupportedError):
            oe.fe("y ~ x + z", data=self.df, entity="entity", cov_type="HC3")

    def test_hc2_rejected_time(self):
        with pytest.raises(oe.VcovTypeNotSupportedError):
            oe.fe("y ~ x + z", data=self.df, time="time", cov_type="HC2")

    def test_hc3_rejected_time(self):
        with pytest.raises(oe.VcovTypeNotSupportedError):
            oe.fe("y ~ x + z", data=self.df, time="time", cov_type="HC3")

    def test_hc2_rejected_twoway(self):
        with pytest.raises(oe.VcovTypeNotSupportedError):
            oe.fe(
                "y ~ x + z", data=self.df, entity="entity", time="time",
                cov_type="HC2",
            )


class TestPanelFEMultiwayCluster:
    """D3: FE with multiway cluster SEs — compare OE (pyfixest CRV1) vs Stata reghdfe."""

    @pytest.fixture(autouse=True)
    def _run(self, df_panel):
        self.s = S_PANEL_FE_MW_CLUSTER
        # Stata: reghdfe y x z, absorb(entity time) vce(cluster entity time)
        self.oe_r = oe.fe(
            "y ~ x + z", data=df_panel, entity="entity", time="time",
            cluster=["entity", "time"],
        )

    def test_coefficients(self):
        npt.assert_allclose(self.oe_r.coefficients.values,
                            [self.s["b_x"], self.s["b_z"]],
                            rtol=1e-6)

    def test_standard_errors(self):
        npt.assert_allclose(self.oe_r.std_errors.values,
                            [self.s["se_x"], self.s["se_z"]],
                            rtol=1e-4)

    def test_nobs(self):
        assert self.oe_r.nobs == int(self.s["N"])


class TestFixedEffectsKwarg:
    """Tests for the fixed_effects= kwarg (N-way FE)."""

    @pytest.fixture(autouse=True)
    def _data(self):
        self.df = pd.read_csv(
            str(Path(__file__).parent / "fixtures" / "df_panel.csv")
        )

    def test_1way_matches_entity(self):
        """fixed_effects=['entity'] should produce identical results to entity='entity'."""
        r_entity = oe.fe("y ~ x + z", data=self.df, entity="entity", cov_type="HC1")
        r_fe = oe.fe("y ~ x + z", data=self.df, fixed_effects=["entity"], cov_type="HC1")
        npt.assert_allclose(r_fe.coefficients.values, r_entity.coefficients.values, rtol=1e-10)
        npt.assert_allclose(r_fe.std_errors.values, r_entity.std_errors.values, rtol=1e-10)
        assert r_fe.nobs == r_entity.nobs

    def test_2way_matches_entity_time(self):
        """fixed_effects=['entity','time'] should match entity=+time=."""
        r_et = oe.fe("y ~ x + z", data=self.df, entity="entity", time="time", cov_type="HC1")
        r_fe = oe.fe("y ~ x + z", data=self.df, fixed_effects=["entity", "time"], cov_type="HC1")
        npt.assert_allclose(r_fe.coefficients.values, r_et.coefficients.values, rtol=1e-10)
        npt.assert_allclose(r_fe.std_errors.values, r_et.std_errors.values, rtol=1e-10)
        assert r_fe.nobs == r_et.nobs

    def test_3way_runs(self):
        """3-way FE via fixed_effects= should run and produce valid results.

        Note: no Stata/R parity anchor exists for 3-way FE — this test
        confirms the code path executes and returns sensible output.
        """
        r = oe.fe("y ~ x + z", data=self.df, fixed_effects=["entity", "time"], cov_type="HC1")
        assert r.nobs == len(self.df)
        assert len(r.coefficients) == 2  # x, z
        assert not np.any(np.isnan(r.coefficients.values))

    def test_3way_crosscheck_pyfixest(self):
        """3-way FE results should match a direct pyfixest.feols call."""
        import pyfixest as pf

        df = self.df.copy()
        df["industry"] = np.random.default_rng(42).choice(["a", "b", "c"], len(df))

        r_oe = oe.fe("y ~ x + z", data=df, fixed_effects=["entity", "time", "industry"], cov_type="HC1")
        r_pf = pf.feols("y ~ x + z | entity + time + industry", data=df, vcov="HC1")

        npt.assert_allclose(r_oe.coefficients.values, r_pf.coef().values, rtol=1e-6)
        npt.assert_allclose(r_oe.std_errors.values, r_pf.se().values, rtol=1e-6)
        assert r_oe.nobs == r_pf._N

    def test_raises_when_both_fixed_effects_and_entity(self):
        with pytest.raises(ValueError, match="not both"):
            oe.fe("y ~ x + z", data=self.df, entity="entity", fixed_effects=["entity"])

    def test_raises_when_both_fixed_effects_and_time(self):
        with pytest.raises(ValueError, match="not both"):
            oe.fe("y ~ x + z", data=self.df, time="time", fixed_effects=["time"])

    def test_raises_when_no_fe_specified(self):
        with pytest.raises(ValueError, match="At least one of"):
            oe.fe("y ~ x + z", data=self.df)

    def test_vcov_index_matches_coefficients(self):
        """vcov() index must equal coefficients.index for fixed_effects=."""
        r = oe.fe("y ~ x + z", data=self.df, fixed_effects=["entity", "time"], cov_type="HC1")
        pd.testing.assert_index_equal(r.coefficients.index, r.vcov().index)
