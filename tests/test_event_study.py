import numpy as np
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
