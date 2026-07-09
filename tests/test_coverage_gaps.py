import warnings

import numpy as np
import pandas as pd
import pytest

import open_econs as oe


class TestNaNHandling:
    def test_drop_warning(self):
        df = pd.DataFrame({
            "y": [1.0, 2.0, np.nan, 4.0],
            "x": [10.0, 20.0, 30.0, np.nan],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = oe.ols("y ~ x", data=df)
            assert len(w) >= 1
            assert any("Dropped" in str(m.message) for m in w)
        assert r.nobs == 2

    def test_all_nan_drops(self):
        df = pd.DataFrame({
            "y": [np.nan, np.nan],
            "x": [np.nan, np.nan],
        })
        with pytest.raises(ValueError, match="0 rows"):
            oe.ols("y ~ x", data=df)

    def test_partial_nan_works(self):
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0, np.nan],
            "x": [10.0, np.nan, 30.0, 40.0],
        })
        r = oe.ols("y ~ x", data=df)
        assert r.nobs == 2


class TestCollinearity:
    def test_perfect_collinearity_raises_or_warns(self, df_ols):
        """Rank-deficient design matrix should raise singular_matrix_error."""
        df = df_ols.copy()
        df["x_dup"] = df["education"] * 1.0
        with pytest.raises(RuntimeError, match="Singular design matrix"):
            oe.ols("income ~ education + age + x_dup", data=df)

    def test_singular_cluster_column(self, df_ols):
        with pytest.raises(ValueError, match="Cluster column"):
            oe.ols("income ~ education + age", data=df_ols, cluster="nonexistent")


class TestExtremeShapes:
    def test_single_observation_raises_singular(self):
        df = pd.DataFrame({"y": [5.0], "x": [2.0]})
        with pytest.raises(RuntimeError, match="Singular design matrix"):
            oe.ols("y ~ x", data=df)

    def test_zero_rows(self):
        df = pd.DataFrame({"y": [], "x": []})
        # formulaic will raise on empty data
        with pytest.raises((ValueError, TypeError)):
            oe.ols("y ~ x", data=df)

    def test_two_observations(self):
        df = pd.DataFrame({"y": [1.0, 3.0], "x": [2.0, 4.0]})
        r = oe.ols("y ~ x", data=df)
        assert r.nobs == 2
        assert r.df_resid == 0


class TestOaxacaEdgeCases:
    def test_by_not_in_xx_raises(self, df_oaxaca):
        """by column is in data but not in the formula RHS."""
        with pytest.raises(ValueError, match="not found in the design matrix"):
            oe.oaxaca("income ~ education + age", data=df_oaxaca, by="female")

    def test_single_observation_per_group(self):
        df = pd.DataFrame({
            "y": [10.0, 20.0],
            "x": [1.0, 5.0],
            "g": [0.0, 1.0],
        })
        d = oe.oaxaca("y ~ x + g", data=df, by="g")
        assert d.nobs == 2

    def test_three_fold_gap_consistent_across_types(self, df_oaxaca):
        d2 = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            decomposition_type="two-fold",
        )
        d3 = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            decomposition_type="three-fold",
        )
        assert abs(d2.total_gap - d3.total_gap) < 1e-10

    def test_group_labels_are_strings(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
        )
        for g in d.by_groups:
            assert isinstance(g, str)


class TestOaxacaStdErrorsNotImplemented:
    """The std field is None in v0.1 but the tidy path for non-None std
    needs a coverage exercise."""

    def test_tidy_without_std(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
        )
        tidy = d.tidy()
        assert list(tidy.columns) == ["Component", "Effect"]

    def test_std_is_none(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
        )
        assert d.std is None


class TestStringColumns:
    def test_categorical_in_formula(self, df_ols):
        r = oe.ols("income ~ education + age + C(province)", data=df_ols)
        assert r.nobs == len(df_ols)
        assert r.coefficients is not None

    def test_categorical_in_oaxaca(self):
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [10.0, 20.0, 30.0, 40.0],
            "g": ["a", "a", "b", "b"],
        })
        # formulaic auto-detects strings and encodes them as "g[T.b]";
        # our resolver adds the original binary column back to XX
        d = oe.oaxaca("y ~ x + g", data=df, by="g")
        assert d.nobs == 4


class TestBaseModelStubs:
    def test_plot_raises_on_all_results(self, df_ols):
        r = oe.ols("income ~ education + age", data=df_ols)
        with pytest.raises(NotImplementedError, match="plot"):
            r.plot()

    def test_export_md_raises(self, df_ols, tmp_path):
        r = oe.ols("income ~ education + age", data=df_ols)
        with pytest.raises(NotImplementedError):
            r.export(str(tmp_path / "out.md"))


class TestContextRepeatedUse:
    def test_multiple_calls(self, df_ols):
        ctx = oe.Context(df_ols)
        r1 = ctx.ols("income ~ education + age")
        r2 = ctx.ols("income ~ education")
        assert r1.nobs == r2.nobs
        assert len(r1.coefficients) != len(r2.coefficients)