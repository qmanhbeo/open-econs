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