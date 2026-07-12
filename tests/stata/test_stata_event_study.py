"""Stata parity tests for event-study regression."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import read_stata

pytestmark = pytest.mark.stata

S_ES = read_stata("event_study")

FIXTURE_CSV = "tests/stata/fixtures/df_event_study.csv"


@pytest.fixture(scope="module")
def df_es() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_CSV)
    df["treated_event_time"] = np.where(df["treated"] == 1, df["post"] - 1, np.nan)
    return df


class TestEventStudyParity:
    """Compare oe.event_study() output against Stata fixture at rtol=1e-6.

    N and df_r are asserted *before* coefficient/sigma checks: if the sample
    differs, no downstream comparison is meaningful.
    """

    @pytest.fixture(autouse=True)
    def _run(self, df_es):
        self.s = S_ES
        self.df = df_es

    # ------------------------------------------------------------------
    # Model 1: no covariates
    # ------------------------------------------------------------------
    def _assert_model(self, prefix: str, oe_r, map_coef: dict[str, str]):
        """Assert N and df_r match, then compare each mapped coefficient.

        Parameters
        ----------
        prefix : str
            Stata scalar prefix, e.g. ``"m1_"`` or ``"m2_"``.
        oe_r : EventStudyResult
        map_coef : dict[str, str]
            Maps OE coefficient name → Stata scalar suffix, e.g.
            ``{"Intercept": "Intercept", "C(…)[T.0.0]": "post"}``.
        """
        # N and df_r match first — fail loud if sample differs
        assert oe_r.nobs == int(self.s[f"{prefix}N"])
        npt.assert_equal(oe_r.nobs - len(oe_r.coefficients), int(self.s[f"{prefix}df_r"]))

        # R²
        npt.assert_allclose(oe_r.r_squared, self.s[f"{prefix}r2"], rtol=1e-6)

        for oe_name, stata_suffix in map_coef.items():
            if "event_cat" in oe_name:
                # Match by period label in oe_r.event_coefficients
                ev = oe_r.event_coefficients.set_index("period")
                period = self._period_from_oe_name(oe_name)
                assert period in ev.index, f"Period {period} not found in event_coefficients"
                oe_coef = ev.loc[period, "coef"]
                oe_ci_l = ev.loc[period, "ci_lower"]
                oe_ci_u = ev.loc[period, "ci_upper"]
                # SE/t/p live in the flat coefficient arrays
                oe_se = oe_r.std_errors[oe_name]
                oe_t = oe_r.t_stats[oe_name]
                oe_p = oe_r.p_values[oe_name]
            else:
                oe_coef = oe_r.coefficients[oe_name]
                oe_se = oe_r.std_errors[oe_name]
                oe_t = oe_r.t_stats[oe_name]
                oe_p = oe_r.p_values[oe_name]
                oe_ci_l, oe_ci_u = self._ci_for_coef(oe_r, oe_name)

            # Compare coefficient, SE, t-stat at strict tolerance
            npt.assert_allclose(oe_coef, self.s[f"{prefix}coef_{stata_suffix}"], rtol=1e-6)
            npt.assert_allclose(oe_se, self.s[f"{prefix}se_{stata_suffix}"], rtol=1e-6)
            npt.assert_allclose(oe_t, self.s[f"{prefix}t_{stata_suffix}"], rtol=1e-6)
            # p-values in extreme tails amplify small t-differences; use relaxed tol
            npt.assert_allclose(oe_p, self.s[f"{prefix}p_{stata_suffix}"], rtol=1e-4)
            # CI lower/upper
            npt.assert_allclose(oe_ci_l, self.s[f"{prefix}ci95l_{stata_suffix}"], rtol=1e-6)
            npt.assert_allclose(oe_ci_u, self.s[f"{prefix}ci95u_{stata_suffix}"], rtol=1e-6)

    @staticmethod
    def _period_from_oe_name(name: str) -> float:
        """Extract the event period from a formulaic column name like
        ``C(treated_event_cat, Treatment(-1))[T.0.0]``."""
        raw = name.rsplit("[T.", 1)[-1].rstrip("]")
        return float(raw)

    @staticmethod
    def _ci_for_coef(oe_r, name: str) -> tuple[float, float]:
        """Return (lower, upper) 95% CI from the result object for *name*."""
        ci = oe_r.conf_int
        return float(ci.loc[name, "lower"]), float(ci.loc[name, "upper"])

    # ------------------------------------------------------------------
    # Test methods
    # ------------------------------------------------------------------
    def _run_m1(self):
        return oe.event_study(
            "y ~ treated * post", data=self.df,
            treatment="treated", post="post",
        )

    def _run_m2(self):
        return oe.event_study(
            "y ~ treated * post + x", data=self.df,
            treatment="treated", post="post",
        )

    def test_m1_n_and_df_r(self):
        r = self._run_m1()
        assert r.nobs == int(self.s["m1_N"])
        npt.assert_equal(r.nobs - len(r.coefficients), int(self.s["m1_df_r"]))

    def test_m1_coefficients(self):
        r = self._run_m1()
        # Find the event-time coefficient name
        ev_col = [c for c in r.coefficients.index if "event_cat" in c]
        assert len(ev_col) == 1
        self._assert_model("m1_", r, {
            "Intercept": "Intercept",
            ev_col[0]: "post",
        })

    def test_m2_n_and_df_r(self):
        r = self._run_m2()
        assert r.nobs == int(self.s["m2_N"])
        npt.assert_equal(r.nobs - len(r.coefficients), int(self.s["m2_df_r"]))

    def test_m2_coefficients(self):
        r = self._run_m2()
        ev_col = [c for c in r.coefficients.index if "event_cat" in c]
        assert len(ev_col) == 1
        self._assert_model("m2_", r, {
            "Intercept": "Intercept",
            ev_col[0]: "post",
            "x": "x",
        })
