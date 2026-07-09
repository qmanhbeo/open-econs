import numpy as np
import pandas as pd
import pytest

import open_econs as oe


@pytest.fixture
def df_iv() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    z1 = np.random.normal(0, 1, n)
    z2 = np.random.normal(0, 1, n)
    x = 0.5 + 0.8 * z1 + 0.6 * z2 + np.random.normal(0, 0.5, n)
    y = 1.0 + 0.5 * x + np.random.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})


class TestIV:
    def test_basic_iv(self, df_iv):
        r = oe.iv("y ~ x | z1 + z2", data=df_iv)
        assert isinstance(r.coefficients, pd.Series)
        assert len(r.coefficients) == 2  # intercept, x

    def test_tidy_shape(self, df_iv):
        r = oe.iv("y ~ x | z1 + z2", data=df_iv)
        tidy = r.tidy()
        expected = {"Variable", "Coef", "Std Err", "z", "P>|z|", "0.025", "0.975"}
        assert set(tidy.columns) == expected
        assert len(tidy) == 2

    def test_summary_returns_string(self, df_iv):
        r = oe.iv("y ~ x | z1 + z2", data=df_iv)
        s = r.summary()
        assert isinstance(s, str)
        assert "IV-2SLS" in s

    def test_first_stage_f_nonzero(self, df_iv):
        r = oe.iv("y ~ x | z1 + z2", data=df_iv)
        assert r.first_stage_f > 10
        assert r.first_stage_p_value < 0.05

    def test_first_stage_dataframe(self, df_iv):
        r = oe.iv("y ~ x | z1 + z2", data=df_iv)
        fs = r.first_stage()
        assert isinstance(fs, pd.DataFrame)
        assert "F" in fs.columns

    def test_no_pipe_raises(self, df_iv):
        with pytest.raises(ValueError, match="three-part"):
            oe.iv("y ~ x", data=df_iv)

    def test_nobs_df(self, df_iv):
        r = oe.iv("y ~ x | z1 + z2", data=df_iv)
        assert r.nobs == len(df_iv)

    def test_conf_int(self, df_iv):
        r = oe.iv("y ~ x | z1 + z2", data=df_iv)
        assert (r.conf_int["lower"] < r.conf_int["upper"]).all()

    def test_missing_column_raises(self, df_iv):
        with pytest.raises(ValueError, match="Column.*not found"):
            oe.iv("y ~ nonexistent | z1 + z2", data=df_iv)

    def test_immutability(self, df_iv):
        r = oe.iv("y ~ x | z1 + z2", data=df_iv)
        with pytest.raises(AttributeError, match="immutable"):
            r.new_attr = 42

    def test_export_json(self, df_iv, tmp_path):
        r = oe.iv("y ~ x | z1 + z2", data=df_iv)
        path = tmp_path / "iv_result.json"
        r.export(str(path))
        assert path.exists()