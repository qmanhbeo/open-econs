"""Numerical parity tests for the v0.6 panel-data engine.

Every estimator is cross-checked against an independent implementation:
- FE     -> linearmodels PanelOLS (entity + time effects)
- RE     -> linearmodels RandomEffects (same backend, tests our wrapper)
- pooled -> statsmodels OLS in levels
- FD     -> linearmodels FirstDifferenceOLS
- DK SE  -> linearmodels PooledOLS with cov_type="kernel"
Both a canonical real dataset (Grunfeld) and a deterministic synthetic
panel (fixed seed) are used. These run on every CI push.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import (
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from scipy.linalg import pinv

import open_econs as oe


def _panel_index(df: pd.DataFrame, ent: str, tm: str) -> pd.DataFrame:
    return df.set_index([ent, tm])


def _fe_ref_slopes(lm_fit) -> dict:
    drop = {"Intercept", "EntityEffects", "TimeEffects", "entity_effects",
            "time_effects"}
    return {k: float(v) for k, v in lm_fit.params.items() if k not in drop}


class TestFEvsLinearmodels:
    def test_fe_grunfeld_matches_panelols(self, df_grunfeld):
        pdf = _panel_index(df_grunfeld, "firm", "year")
        lm = PanelOLS.from_formula(
            "invest ~ value + capital + EntityEffects + TimeEffects", pdf
        ).fit()
        r = oe.fe(
            "invest ~ value + capital",
            data=df_grunfeld, entity="firm", time="year",
        )
        ref = _fe_ref_slopes(lm)
        for name, val in ref.items():
            npt.assert_allclose(r.coefficients[name], val, rtol=1e-6, err_msg=name)

    def test_fe_grunfeld_se_matches_within_ols(self, df_grunfeld):
        # oe.fe's within SE uses statsmodels df_resid, so the canonical
        # reference is OLS on the group-demeaned (within) data — not
        # linearmodels' debiased PanelOLS SE.
        r = oe.fe(
            "invest ~ value + capital",
            data=df_grunfeld, entity="firm", time="year", cov_type="nonrobust",
        )
        d = df_grunfeld

        def _demean(v, g):
            s = pd.DataFrame({"v": v, "g": g})
            return v - s.groupby("g")["v"].transform("mean")

        ent, tm = d["firm"].values, d["year"].values
        yd = _demean(d["invest"].values, ent)
        yd = _demean(yd, tm)
        cols = ["value", "capital"]
        Xd = np.column_stack([_demean(_demean(d[c].values, ent), tm) for c in cols])
        # Within transform absorbs the intercept: fit demeaned slopes, no constant.
        ref = sm.OLS(yd, Xd).fit()
        for i, name in enumerate(cols):
            npt.assert_allclose(r.std_errors[name], ref.bse.iloc[i], rtol=1e-9, err_msg=name)

    def test_fe_synthetic_matches_panelols(self, df_panel):
        pdf = _panel_index(df_panel, "entity", "time")
        lm = PanelOLS.from_formula(
            "y ~ x + z + EntityEffects + TimeEffects", pdf
        ).fit()
        r = oe.fe("y ~ x + z", data=df_panel, entity="entity", time="time")
        ref = _fe_ref_slopes(lm)
        for name, val in ref.items():
            npt.assert_allclose(r.coefficients[name], val, rtol=1e-6, err_msg=name)

    def test_fe_unbalanced_matches_panelols(self, df_panel_unbalanced):
        pdf = _panel_index(df_panel_unbalanced, "entity", "time")
        lm = PanelOLS.from_formula(
            "y ~ x + z + EntityEffects + TimeEffects", pdf
        ).fit()
        r = oe.fe("y ~ x + z", data=df_panel_unbalanced, entity="entity", time="time")
        ref = _fe_ref_slopes(lm)
        for name, val in ref.items():
            npt.assert_allclose(r.coefficients[name], val, rtol=1e-6, err_msg=name)


class TestREWrapperParity:
    """RE delegates to linearmodels; these confirm our wrapper extracts right."""

    def test_re_grunfeld_slopes_match_lm(self, df_grunfeld):
        pdf = _panel_index(df_grunfeld, "firm", "year")
        lm = RandomEffects.from_formula("invest ~ 1 + value + capital", pdf).fit()
        r = oe.PanelContext(df_grunfeld, entity="firm", time="year").re(
            "invest ~ value + capital"
        )
        npt.assert_allclose(r.coefficients.values, lm.params.values, rtol=1e-9)

    def test_re_grunfeld_se_match_lm(self, df_grunfeld):
        pdf = _panel_index(df_grunfeld, "firm", "year")
        lm = RandomEffects.from_formula("invest ~ 1 + value + capital", pdf).fit()
        r = oe.PanelContext(df_grunfeld, entity="firm", time="year").re(
            "invest ~ value + capital"
        )
        npt.assert_allclose(r.std_errors.values, lm.std_errors.values, rtol=1e-9)

    def test_re_vcov_matches_lm(self, df_panel):
        pdf = _panel_index(df_panel, "entity", "time")
        lm = RandomEffects.from_formula("y ~ 1 + x + z", pdf).fit()
        r = oe.PanelContext(df_panel, entity="entity", time="time").re("y ~ x + z")
        npt.assert_allclose(
            r.vcov().values,
            lm.cov.loc[r.coefficients.index, r.coefficients.index].values,
            rtol=1e-9,
        )


class TestPooledAndFDParity:
    def test_pooled_matches_statsmodels_ols(self, df_panel):
        r = oe.PanelContext(df_panel, entity="entity", time="time").pooled("y ~ x + z")
        smr = sm.OLS(df_panel["y"], sm.add_constant(df_panel[["x", "z"]])).fit()
        npt.assert_allclose(r.coefficients.values, smr.params.values, rtol=1e-9)
        npt.assert_allclose(r.std_errors.values, smr.bse.values, rtol=1e-9)

    def test_fd_matches_linearmodels(self, df_panel):
        pdf = _panel_index(df_panel, "entity", "time")
        lm = FirstDifferenceOLS.from_formula("y ~ x + z", pdf).fit()
        r = oe.PanelContext(df_panel, entity="entity", time="time").diff("y ~ x + z")
        npt.assert_allclose(r.coefficients.values, lm.params.values, rtol=1e-9)

    def test_dk_se_matches_linearmodels(self, df_panel):
        pdf = _panel_index(df_panel, "entity", "time")
        lm = PooledOLS.from_formula("y ~ 1 + x + z", pdf).fit(cov_type="kernel")
        r = oe.PanelContext(df_panel, entity="entity", time="time").driscoll_kraay("y ~ x + z")
        npt.assert_allclose(r.std_errors.values, lm.std_errors.values, rtol=1e-9)


class TestHausmanParity:
    def test_hausman_grunfeld_valid(self, df_grunfeld):
        pc = oe.PanelContext(df_grunfeld, entity="firm", time="year")
        fe = pc.fe("invest ~ value + capital")
        re = pc.re("invest ~ value + capital")
        h = pc.hausman(fe, re)
        assert 0.0 <= h.statistic
        assert 0.0 <= h.p_value <= 1.0
        assert h.df == 2
        # Grunfeld firm effects are uncorrelated with regressors -> not rejected.
        assert h.rejected_at() is False

    def test_hausman_matches_manual(self, df_panel):
        pc = oe.PanelContext(df_panel, entity="entity", time="time")
        fe = pc.fe("y ~ x + z")
        re = pc.re("y ~ x + z")
        h = pc.hausman(fe, re)
        common = sorted(set(fe.coefficients.index) & set(re.coefficients.index))
        d = fe.coefficients[common].values - re.coefficients[common].values
        V = (fe.vcov().loc[common, common].values
             - re.vcov().loc[common, common].values)
        V = (V + V.T) / 2
        H_manual = float(d @ pinv(V) @ d)
        npt.assert_allclose(h.statistic, max(H_manual, 0.0), rtol=1e-9)
