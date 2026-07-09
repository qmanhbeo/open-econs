import numpy as np
import pandas as pd
import pytest

import open_econs as oe


@pytest.fixture
def df_panel() -> pd.DataFrame:
    np.random.seed(42)
    n_entities = 50
    t_periods = 5
    rows = []
    for e in range(n_entities):
        entity_fe = np.random.normal(0, 2)
        for t in range(t_periods):
            x1 = np.random.uniform(0, 10)
            x2 = np.random.uniform(-5, 5)
            y = entity_fe + 0.8 * x1 + 0.3 * x2 + np.random.normal(0, 1)
            rows.append({
                "entity_id": f"E{e:03d}",
                "year": 2010 + t,
                "y": y,
                "x1": x1,
                "x2": x2,
                "cluster_var": np.random.choice(["A", "B", "C"]),
            })
    return pd.DataFrame(rows)


class TestFE:
    def test_basic_fe(self, df_panel):
        r = oe.fe("y ~ x1 + x2", data=df_panel, entity="entity_id")
        assert isinstance(r.coefficients, pd.Series)
        assert len(r.coefficients) == 2  # x1, x2 (no intercept in within)
        assert r.nobs == len(df_panel)

    def test_two_way_fe(self, df_panel):
        r = oe.fe("y ~ x1 + x2", data=df_panel, entity="entity_id", time="year")
        assert len(r.coefficients) == 2
        assert r.nobs == len(df_panel)

    def test_fe_matches_group_demeaned_ols(self, df_panel):
        y = df_panel["y"].values
        x1 = df_panel["x1"].values
        x2 = df_panel["x2"].values
        entity_ids = df_panel["entity_id"].values
        unique_entities = np.unique(entity_ids)

        y_demeaned = y.copy().astype(float)
        x1_demeaned = x1.copy().astype(float)
        x2_demeaned = x2.copy().astype(float)

        for e in unique_entities:
            mask = entity_ids == e
            y_demeaned[mask] -= y[mask].mean()
            x1_demeaned[mask] -= x1[mask].mean()
            x2_demeaned[mask] -= x2[mask].mean()

        import statsmodels.api as sm
        X = np.column_stack([x1_demeaned, x2_demeaned])
        sm_r = sm.OLS(y_demeaned, X).fit()

        r = oe.fe("y ~ x1 + x2", data=df_panel, entity="entity_id", cov_type="nonrobust")

        np.testing.assert_allclose(r.coefficients.values, sm_r.params, rtol=1e-10)
        np.testing.assert_allclose(r.std_errors.values, sm_r.bse, rtol=1e-8)

    def test_fe_tidy(self, df_panel):
        r = oe.fe("y ~ x1 + x2", data=df_panel, entity="entity_id")
        tidy = r.tidy()
        assert "Variable" in tidy.columns
        assert "Coef" in tidy.columns
        assert len(tidy) == 2

    def test_fe_summary(self, df_panel):
        r = oe.fe("y ~ x1 + x2", data=df_panel, entity="entity_id")
        s = r.summary()
        assert "OLS" in s

    def test_fe_cluster_se(self, df_panel):
        r = oe.fe("y ~ x1 + x2", data=df_panel, entity="entity_id", cluster="cluster_var")
        assert "cluster(cluster_var)" in r.cov_type

    def test_fe_no_entity_raises(self, df_panel):
        with pytest.raises(ValueError, match="At least one"):
            oe.fe("y ~ x1 + x2", data=df_panel)

    def test_fe_in_sample_predict(self, df_panel):
        r = oe.fe("y ~ x1 + x2", data=df_panel, entity="entity_id")
        pred = r.predict()
        assert len(pred) == r.nobs

    def test_fe_conf_int(self, df_panel):
        r = oe.fe("y ~ x1 + x2", data=df_panel, entity="entity_id")
        assert (r.conf_int["lower"] < r.conf_int["upper"]).all()

    def test_fe_data_shape(self, df_panel):
        r = oe.fe("y ~ x1 + x2", data=df_panel, entity="entity_id")
        assert r.data_shape == (len(df_panel), 2)

    def test_fe_export_json(self, df_panel, tmp_path):
        r = oe.fe("y ~ x1 + x2", data=df_panel, entity="entity_id")
        path = tmp_path / "fe_result.json"
        r.export(str(path))
        assert path.exists()