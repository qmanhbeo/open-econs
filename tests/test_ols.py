import numpy as np
import pandas as pd
import pytest

import open_econs as oe


class TestOLS:
    def test_basic_ols(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        assert isinstance(r.coefficients, pd.Series)
        assert len(r.coefficients) == 3  # Intercept, education, age
        assert "Intercept" in r.coefficients.index
        assert "education" in r.coefficients.index
        assert isinstance(r.r_squared, float)
        assert 0 <= r.r_squared <= 1

    def test_nobs_df(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        assert r.nobs == len(df_ols)
        assert r.df_resid == r.nobs - r.df_model - 1

    def test_tidy_shape(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        tidy = r.tidy()
        assert isinstance(tidy, pd.DataFrame)
        expected_cols = {"Variable", "Coef", "Std Err", "t", "P>|t|", "0.025", "0.975"}
        assert set(tidy.columns) == expected_cols
        assert len(tidy) == 3

    def test_clustered_ols(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols, cluster="province")
        assert "cluster(province)" in r.cov_type
        assert r.nobs == len(df_ols)

    def test_reg_alias(self, df_ols):
        r = oe.reg("income ~ education + age", data=df_ols)
        assert r.nobs == len(df_ols)

    def test_predict(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        pred = r.predict(df_ols.head(10))
        assert isinstance(pred, pd.Series)
        assert len(pred) == 10
        assert pred.name == "predicted"

    def test_predict_in_sample(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        pred = r.predict()
        assert len(pred) == r.nobs

    def test_summary_returns_string(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        s = r.summary()
        assert isinstance(s, str)
        assert "OLS Regression" in s

    def test_repr(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        assert isinstance(repr(r), str)

    def test_immutability(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        with pytest.raises(AttributeError, match="immutable"):
            r.new_attr = 42

    def test_export_json(self, df_ols, tmp_path):
        r = oe.ols("income ~ education + age", data=df_ols)
        path = tmp_path / "result.json"
        r.export(str(path))
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert "formula" in data
        assert "results" in data

    def test_export_unsupported_raises(self, df_ols, tmp_path):
        r = oe.ols("income ~ education + age", data=df_ols)
        with pytest.raises(NotImplementedError, match="Only .json"):
            r.export(str(tmp_path / "result.md"))

    def test_plot_raises(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        with pytest.raises(NotImplementedError, match="plot"):
            r.plot()

    def test_nonrobust_se(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols, cov_type="nonrobust")
        assert r.cov_type == "nonrobust"
        assert r.std_errors is not None

    def test_missing_cluster_column(self, df_ols):
        with pytest.raises(ValueError, match="Cluster column"):
            oe.ols("income ~ education + age", data=df_ols, cluster="nonexistent")

    def test_coefficients_are_series_with_names(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        assert isinstance(r.coefficients["education"], np.floating)

    def test_conf_int_structure(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        assert list(r.conf_int.columns) == ["lower", "upper"]
        assert (r.conf_int["lower"] < r.conf_int["upper"]).all()

    def test_no_intercept_formula(self, df_ols):
        r = oe.ols("income ~ education + age - 1", data=df_ols)
        assert "Intercept" not in r.coefficients.index
        assert r.df_model == 2

    def test_predict_newdata_without_lhs(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        test = df_ols[["education", "age"]].iloc[:5]
        pred = r.predict(test)
        assert len(pred) == 5
        assert pred.name == "predicted"

    def test_predict_newdata_no_intercept(self, df_ols):
        r = oe.ols("income ~ education - 1", data=df_ols)
        test = df_ols[["education"]].iloc[:3]
        pred = r.predict(test)
        assert len(pred) == 3

    def test_rhs_formula_stored(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        assert r.rhs_formula == "education + age"

    def test_summary_shows_llf_aic_bic(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        s = r.summary()
        assert "Log-Likelihood" in s
        assert "AIC" in s
        assert "BIC" in s
        assert "N/A" not in s.split("Log-Likelihood:")[1].split("\n")[0]

    def test_package_version(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        from open_econs._version import __version__
        assert r.package_version == __version__

    def test_predict_with_C_factor(self, df_categorical):
        r = oe.ols("income ~ education + C(region)", data=df_categorical)
        pred = r.predict(df_categorical.head(5))
        assert len(pred) == 5
        assert isinstance(pred, pd.Series)

    def test_predict_C_factor_newdata_subset(self, df_categorical):
        r = oe.ols("income ~ education + C(region)", data=df_categorical)
        test = df_categorical[["education", "region"]].head(3)
        pred = r.predict(test)
        assert len(pred) == 3

    def test_collinearity_warns(self, df_collinear):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            oe.ols("y ~ x + x_dup", data=df_collinear)
        assert any("near-singular" in str(msg.message).lower() for msg in w)

    def test_robust_f_statistic_differs_from_homoskedastic(self, df_ols):
        r_nr = oe.ols("income ~ education + age", data=df_ols, cov_type="nonrobust")
        r_hc1 = oe.ols("income ~ education + age", data=df_ols, cov_type="HC1")
        assert r_nr.f_statistic != r_hc1.f_statistic or abs(r_nr.f_p_value - r_hc1.f_p_value) > 1e-12

    def test_tidy_conf_int_columns_no_brackets(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        tidy = r.tidy()
        assert "0.025" in tidy.columns
        assert "0.975" in tidy.columns
        assert "[0.025" not in tidy.columns
        assert "0.975]" not in tidy.columns