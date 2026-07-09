import numpy as np
import pandas as pd
import pytest

import open_econs as oe


class TestOaxaca:
    def test_two_fold(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        assert isinstance(d.explained, float)
        assert isinstance(d.unexplained, float)
        assert isinstance(d.total_gap, float)
        assert d.type == "two-fold"

    def test_three_fold(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca,
            by="female",
            decomposition_type="three-fold",
        )
        assert d.type == "three-fold"
        assert isinstance(d.total_gap, float)

    def test_tidy_two_fold(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        tidy = d.tidy()
        assert list(tidy["Component"]) == ["Explained", "Unexplained", "Total Gap"]
        assert list(tidy.columns) == ["Component", "Effect"]

    def test_tidy_three_fold(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca,
            by="female",
            decomposition_type="three-fold",
        )
        tidy = d.tidy()
        assert list(tidy["Component"]) == ["Endowment", "Coefficients", "Interaction", "Total Gap"]

    def test_gap_consistent(self, df_oaxaca):
        d2 = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        d3 = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca,
            by="female",
            decomposition_type="three-fold",
        )
        assert abs(d2.total_gap - d3.total_gap) < 1e-10

    def test_nobs(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        assert d.nobs == len(df_oaxaca)

    def test_immutability(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        with pytest.raises(AttributeError, match="immutable"):
            d.new_attr = 42

    def test_predict_raises(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        with pytest.raises(NotImplementedError, match="does not support predict"):
            d.predict()

    def test_summary_returns_string(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        s = d.summary()
        assert isinstance(s, str)
        assert "Oaxaca-Blinder" in s

    def test_export_json(self, df_oaxaca, tmp_path):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        path = tmp_path / "decomp.json"
        d.export(str(path))
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert "formula" in data
        assert "results" in data

    def test_by_groups(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        assert len(d.by_groups) == 2
        assert d.by_groups[0] in ("0.0", "1.0")
        assert d.by_groups[1] in ("0.0", "1.0")

    def test_by_groups_labels_match_data(self, df_oaxaca):
        df = df_oaxaca.copy()
        df["female"] = np.random.choice(["Male", "Female"], len(df))
        d = oe.oaxaca("income ~ education + age + female", data=df, by="female")
        assert "Male" in d.by_groups or "Female" in d.by_groups

    def test_swap_ensures_positive_gap(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female", swap=True)
        assert d.total_gap >= 0

    def test_data_shape(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        assert d.data_shape[0] == len(df_oaxaca)
        assert d.data_shape[1] == 4  # Intercept + education + age + female

    def test_invalid_decomposition_type(self, df_oaxaca):
        with pytest.raises(ValueError, match="Unknown"):
            oe.oaxaca(
                "income ~ education + age + female",
                data=df_oaxaca,
                by="female",
                decomposition_type="invalid",
            )

    def test_non_binary_by(self, df_oaxaca):
        df = df_oaxaca.copy()
        df["three"] = np.random.choice([1, 2, 3], size=len(df))
        with pytest.raises(ValueError, match="must be binary"):
            oe.oaxaca("income ~ education + age + three", data=df, by="three")

    def test_missing_by_column(self, df_oaxaca):
        with pytest.raises(ValueError, match="not found"):
            oe.oaxaca("income ~ education + age", data=df_oaxaca, by="nonexistent")

    def test_three_fold_interaction_stored(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            decomposition_type="three-fold",
        )
        assert isinstance(d.interaction, float)

    def test_three_fold_components_sum_to_gap(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            decomposition_type="three-fold",
        )
        total = d.explained + d.unexplained + d.interaction
        assert abs(total - d.total_gap) < 1e-10

    def test_two_fold_std_propagates(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            std=True, bootstrap_n=50, seed=42,
        )
        assert d.std is not None
        assert isinstance(d.std, pd.Series)
        assert len(d.std) == 2  # unexplained, explained

    def test_three_fold_std_propagates(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            decomposition_type="three-fold",
            std=True, bootstrap_n=50, seed=42,
        )
        assert d.std is not None
        assert len(d.std) == 3  # endowment, coefficients, interaction

    def test_bootstrap_reproducible_with_seed(self, df_oaxaca):
        d1 = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            std=True, bootstrap_n=100, seed=99,
        )
        d2 = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            std=True, bootstrap_n=100, seed=99,
        )
        assert d1.std is not None and d2.std is not None
        import numpy.testing as npt
        npt.assert_allclose(d1.std.values, d2.std.values, rtol=1e-10)

    def test_bootstrap_different_seed_gives_different(self, df_oaxaca):
        d1 = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            std=True, bootstrap_n=50, seed=42,
        )
        d2 = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            std=True, bootstrap_n=50, seed=7,
        )
        assert d1.std is not None and d2.std is not None
        assert not (d1.std.values == d2.std.values).all()

    def test_tidy_with_std_has_std_err_column(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            std=True, bootstrap_n=50, seed=42,
        )
        tidy = d.tidy()
        assert "Std Err" in tidy.columns

    def test_interaction_in_tidy_three_fold(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            decomposition_type="three-fold",
        )
        tidy = d.tidy()
        interaction_row = tidy[tidy["Component"] == "Interaction"]
        assert not interaction_row.empty
        assert interaction_row["Effect"].values[0] != 0.0 or abs(
            d.explained + d.unexplained + d.interaction - d.total_gap
        ) < 1e-10

    def test_tidy_without_std_has_no_std_err(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            std=False,
        )
        tidy = d.tidy()
        assert "Std Err" not in tidy.columns

    def test_variable_detail_stored(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        assert hasattr(d, "variable_detail")
        assert not d.variable_detail.empty
        assert "Variable" in d.variable_detail.columns
        assert "Explained" in d.variable_detail.columns
        assert "Unexplained" in d.variable_detail.columns

    def test_variable_detail_three_fold(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            decomposition_type="three-fold",
        )
        assert "Endowment" in d.variable_detail.columns
        assert "Coefficients" in d.variable_detail.columns
        assert "Interaction" in d.variable_detail.columns

    def test_variable_detail_rows_match_variables(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        expected_vars = {"Intercept", "education", "age"}
        assert set(d.variable_detail["Variable"]) == expected_vars

    def test_variable_detail_sums_match_aggregate_two_fold(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        explained_sum = d.variable_detail["Explained"].sum()
        unexplained_sum = d.variable_detail["Unexplained"].sum()
        assert abs(explained_sum - d.explained) < 1e-10
        assert abs(unexplained_sum - d.unexplained) < 1e-10

    def test_variable_detail_sums_match_aggregate_three_fold(self, df_oaxaca):
        d = oe.oaxaca(
            "income ~ education + age + female",
            data=df_oaxaca, by="female",
            decomposition_type="three-fold",
        )
        endow_sum = d.variable_detail["Endowment"].sum()
        coeff_sum = d.variable_detail["Coefficients"].sum()
        inter_sum = d.variable_detail["Interaction"].sum()
        assert abs(endow_sum - d.explained) < 1e-10
        assert abs(coeff_sum - d.unexplained) < 1e-10
        assert abs(inter_sum - d.interaction) < 1e-10

    def test_tidy_detail_returns_var_detail(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        detail_tidy = d.tidy(detail=True)
        assert "Variable" in detail_tidy.columns
        assert len(detail_tidy) == len(d.variable_detail)

    def test_tidy_detail_false_returns_aggregate(self, df_oaxaca):
        d = oe.oaxaca("income ~ education + age + female", data=df_oaxaca, by="female")
        agg = d.tidy(detail=False)
        assert "Component" in agg.columns
        assert "Variable" not in agg.columns