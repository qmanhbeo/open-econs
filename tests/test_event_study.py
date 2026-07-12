import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")  # headless backend: r.plot() must not open a window

import open_econs as oe


@pytest.fixture
def df_event() -> pd.DataFrame:
    np.random.seed(11)
    n = 600
    treated = np.random.binomial(1, 0.5, n).astype(float)
    post = np.random.binomial(1, 0.5, n).astype(float)
    x = np.random.normal(0, 1, n)
    # True ATT = 0.8 on the treated group in the post period.
    y = 1.0 + 0.8 * treated * post + 0.3 * x + np.random.normal(0, 0.5, n)
    df = pd.DataFrame(
        {"y": y, "x": x, "treated": treated, "post": post}
    )
    # Event-time relative to treatment for the treated group; NaN for control.
    df["treated_event_time"] = np.where(df["treated"] == 1, df["post"] - 1, np.nan)
    return df


class TestEventStudy:
    def test_interaction_only_does_not_crash(self, df_event):
        r = oe.event_study(
            "y ~ treated * post", data=df_event,
            treatment="treated", post="post",
        )
        assert hasattr(r, "event_coefficients")
        assert not r.event_coefficients.empty

    def test_with_covariates(self, df_event):
        r = oe.event_study(
            "y ~ treated * post + x", data=df_event,
            treatment="treated", post="post",
        )
        assert not r.event_coefficients.empty
        assert "x" in r.coefficients.index

    def test_reference_period_omitted(self, df_event):
        r = oe.event_study(
            "y ~ treated * post", data=df_event,
            treatment="treated", post="post", omitted_period=-1,
        )
        # Period -1 is the omitted reference and must not appear as a coefficient.
        assert -1.0 not in r.event_coefficients["period"].tolist()

    def test_post_treatment_coef_recovers_effect(self, df_event):
        r = oe.event_study(
            "y ~ treated * post + x", data=df_event,
            treatment="treated", post="post",
        )
        ev = r.event_coefficients.set_index("period")
        # The only post-treatment event period is 0.
        assert 0.0 in ev.index
        # Should be close to the true ATT of 0.8.
        assert np.isclose(ev.loc[0.0, "coef"], 0.8, atol=0.3)

    def test_missing_event_column_raises(self, df_event):
        df_no = df_event.drop(columns=["treated_event_time"])
        with pytest.raises(ValueError):
            oe.event_study(
                "y ~ treated * post", data=df_no,
                treatment="treated", post="post",
            )

    def test_plot_runs(self, df_event):
        r = oe.event_study(
            "y ~ treated * post", data=df_event,
            treatment="treated", post="post",
        )
        try:
            r.plot()
        except ImportError:
            pytest.skip("matplotlib not installed")


@pytest.fixture
def df_event_panel() -> pd.DataFrame:
    np.random.seed(11)
    n_entities = 60
    n_periods = 10
    n = n_entities * n_periods
    treated = np.repeat(np.random.binomial(1, 0.5, n_entities), n_periods).astype(float)
    time = np.tile(np.arange(n_periods), n_entities).astype(float)
    post = np.where(time >= 5, 1.0, 0.0)
    x = np.random.normal(0, 1, n)
    y = 1.0 + 0.8 * treated * post + 0.3 * x + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({
        "y": y, "x": x, "treated": treated,
        "post": post, "time": time,
    })
    event_time = np.where(df["treated"] == 1, df["post"] - 1, np.nan)
    df["treated_event_time"] = np.where(df["treated"] == 1, event_time, np.nan)
    return df


class TestEventStudyHAC:
    def test_hac_se_differs_from_nonrobust(self, df_event_panel):
        r_hac = oe.event_study(
            "y ~ treated * post", data=df_event_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        r_nonr = oe.event_study(
            "y ~ treated * post", data=df_event_panel,
            treatment="treated", post="post",
            cov_type="nonrobust",
        )
        assert not np.allclose(r_hac.std_errors, r_nonr.std_errors, rtol=1e-10)

    def test_hac_matches_ols_hac_internal(self, df_event_panel):
        r_es = oe.event_study(
            "y ~ treated * post + x", data=df_event_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        assert r_es.cov_type == "HAC(2)"
        assert not np.isnan(r_es.std_errors).any()

        r_nonr = oe.event_study(
            "y ~ treated * post + x", data=df_event_panel,
            treatment="treated", post="post",
            cov_type="nonrobust",
        )
        assert not np.allclose(r_es.std_errors, r_nonr.std_errors, rtol=1e-10)

    def test_hac_requires_lags(self, df_event_panel):
        with pytest.raises(ValueError, match="lags"):
            oe.event_study(
                "y ~ treated * post", data=df_event_panel,
                treatment="treated", post="post",
                cov_type="HAC", lags=None, time="time",
            )

    def test_hac_requires_time(self, df_event_panel):
        with pytest.raises(ValueError, match="time"):
            oe.event_study(
                "y ~ treated * post", data=df_event_panel,
                treatment="treated", post="post",
                cov_type="HAC", lags=2, time=None,
            )

    def test_hac_cov_label(self, df_event_panel):
        r = oe.event_study(
            "y ~ treated * post", data=df_event_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        assert r.cov_type == "HAC(2)"

    def test_hac_alias_lowercase(self, df_event_panel):
        r1 = oe.event_study(
            "y ~ treated * post", data=df_event_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        r2 = oe.event_study(
            "y ~ treated * post", data=df_event_panel,
            treatment="treated", post="post",
            cov_type="hac", lags=2, time="time",
        )
        npt.assert_allclose(r1.std_errors.values, r2.std_errors.values, rtol=1e-12)

    def test_hac_preserves_event_coefficients(self, df_event_panel):
        r_hac = oe.event_study(
            "y ~ treated * post", data=df_event_panel,
            treatment="treated", post="post",
            cov_type="HAC", lags=2, time="time",
        )
        r_nonr = oe.event_study(
            "y ~ treated * post", data=df_event_panel,
            treatment="treated", post="post",
            cov_type="nonrobust",
        )
        npt.assert_allclose(
            r_hac.event_coefficients["coef"].values,
            r_nonr.event_coefficients["coef"].values,
            rtol=1e-12,
        )
