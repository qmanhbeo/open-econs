import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import open_econs as oe


@pytest.fixture
def df_panel() -> pd.DataFrame:
    np.random.seed(7)
    n_unit = 150
    n_time = 8
    n = n_unit * n_time
    entity = np.repeat(np.arange(n_unit), n_time)
    time = np.tile(np.arange(n_time), n_unit)
    alpha = np.random.normal(0, 2, n_unit)
    beta_t = np.random.normal(0, 1.5, n_time)
    x = np.random.normal(0, 1, n)
    z = np.random.normal(0, 1, n)
    y = 1.5 * x - 0.7 * z + alpha[entity] + beta_t[time] + np.random.normal(0, 0.5, n)
    return pd.DataFrame(
        {"y": y, "x": x, "z": z, "entity": entity, "time": time}
    )


def _demean(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    s = pd.DataFrame({"v": v, "g": g})
    return v - s.groupby("g")["v"].transform("mean")


def _ref_within(y, X, entity, time):
    """Group-demeaned (two-way within) OLS as an independent reference."""
    yd = _demean(_demean(y, entity), time)
    cols = []
    for j in range(X.shape[1]):
        c = _demean(_demean(X[:, j], entity), time)
        cols.append(c)
    Xd = np.column_stack(cols)
    Xd = sm.add_constant(Xd)
    fit = sm.OLS(yd, Xd).fit()
    return fit


class TestFixedEffects:
    def test_coef_matches_reference(self, df_panel):
        r = oe.fe("y ~ x + z", data=df_panel, entity="entity", time="time")
        fit = _ref_within(
            df_panel["y"].values,
            df_panel[["x", "z"]].values,
            df_panel["entity"].values,
            df_panel["time"].values,
        )
        # Intercept is absorbed by FE; compare slopes only.
        ref = fit.params[1:]
        for name, coef in r.coefficients.items():
            if name == "Intercept":
                continue
            idx = list(df_panel[["x", "z"]].columns).index(name)
            assert np.isclose(coef, ref.iloc[idx], atol=1e-6), name

    def test_r_squared_matches_reference(self, df_panel):
        r = oe.fe("y ~ x + z", data=df_panel, entity="entity", time="time")
        fit = _ref_within(
            df_panel["y"].values,
            df_panel[["x", "z"]].values,
            df_panel["entity"].values,
            df_panel["time"].values,
        )
        assert np.isclose(r.r_squared, float(fit.rsquared), atol=1e-6)

    def test_df_resid_consistent_with_standard_errors(self, df_panel):
        r = oe.fe("y ~ x + z", data=df_panel, entity="entity", time="time")
        # The reported df_resid must match the degrees of freedom that
        # statsmodels used to compute the reported standard errors / rsd,
        # otherwise rsd and std_errors disagree.
        assert r.df_resid == int(r._fit.df_resid)
        assert np.isclose(r.rsd, float(np.sqrt(r._fit.scale)), atol=1e-9)

    def test_one_way_fe_consistency(self, df_panel):
        r = oe.fe("y ~ x + z", data=df_panel, entity="entity")
        fit = _ref_within(
            df_panel["y"].values,
            df_panel[["x", "z"]].values,
            df_panel["entity"].values,
            np.zeros(len(df_panel)),
        )
        assert np.isclose(r.r_squared, float(fit.rsquared), atol=1e-6)
        assert r.df_resid == int(r._fit.df_resid)

    def test_no_entity_or_time_raises(self):
        df = pd.DataFrame({"y": [1.0, 2.0], "x": [1.0, 2.0]})
        with pytest.raises(ValueError):
            oe.fe("y ~ x", data=df)
