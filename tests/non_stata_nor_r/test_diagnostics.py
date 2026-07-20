"""Non-Stata/non-R consistency tests for the v1.3 OLS diagnostics battery.

These tests verify the ``open_econs.core.diagnostics`` implementations against
the underlying reference math and, where the implementation wraps
``statsmodels`` functions, against those functions directly. They are NOT
Stata/R parity tests (those live under ``tests/stata`` and ``tests/r``).

All numeric cross-checks use the rule-2 parity tolerance of 1e-6.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox as sm_ljungbox
from statsmodels.stats.outliers_influence import OLSInfluence

import open_econs as oe


def _make_data(seed: int = 0, n: int = 200):
    """Deterministic synthetic OLS dataset: y = X @ beta + e.

    Returns ``(df, X_design, y, beta)`` where ``X_design`` is the full design
    matrix WITH the intercept column (shape n x 3) so a statsmodels
    ``.fit()`` and the from-scratch numpy reconstructions line up exactly.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    beta = np.array([1.0, 2.0, -1.5])
    X = np.column_stack([np.ones(n), x1, x2])
    e = rng.standard_normal(n)
    y = X @ beta + e
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return df, X, y, beta


def _n_r2(resid, Z):
    """Auxiliary-regression LM = n * R^2 from a numpy fit."""
    beta = np.linalg.lstsq(Z, resid, rcond=None)[0]
    fitted = Z @ beta
    ssr = float(np.sum((resid - fitted) ** 2))
    ssr0 = float(np.sum((resid - resid.mean()) ** 2))
    return Z.shape[0] * (ssr0 - ssr) / ssr0


class TestDiagnostics:
    def setup_method(self):
        self.df, self.X, self.y, self.beta = _make_data(seed=0, n=200)
        self.r = oe.ols("y ~ x1 + x2", data=self.df)
        self.sm_fit = sm.OLS(self.y, self.X).fit()
        self.resid = self.r.residuals.values.ravel()

    def test_bg_test_auxiliary_n_r2(self):
        """bg_test LM equals n*R^2 of the rebuilt auxiliary regression."""
        lags = 1
        res = self.resid
        n = len(res)
        L = np.zeros((n, lags))
        for j in range(1, lags + 1):
            col = np.zeros(n)
            col[j:] = res[: n - j]
            L[:, j - 1] = col
        Z = np.column_stack([self.X, L])
        lm = _n_r2(res, Z)
        out = self.r.bg_test(lags=lags)
        assert out["lm_stat"] == pytest.approx(lm, rel=1e-6)
        assert out["df"] == lags
        assert 0.0 < out["lm_pvalue"] < 1.0

    def test_white_test_auxiliary(self):
        """white_test LM equals n*R^2 of the rebuilt White auxiliary; df = 5."""
        res = self.resid
        Xc = self.X[:, 1:]  # drop constant
        n, p = Xc.shape
        cols = [Xc[:, j] for j in range(p)]
        cols += [Xc[:, j] ** 2 for j in range(p)]
        for a in range(p):
            for b in range(a + 1, p):
                cols.append(Xc[:, a] * Xc[:, b])
        Z = np.column_stack(cols) - np.column_stack(cols).mean(axis=0)
        Z = np.column_stack([np.ones(n), Z])
        lm = _n_r2(res ** 2, Z)
        out = self.r.white_test(interaction=True)
        assert out["white_stat"] == pytest.approx(lm, rel=1e-6)
        assert out["df"] == 5.0
        assert 0.0 < out["white_pvalue"] < 1.0

    def test_ljung_box_vs_statsmodels(self):
        """ljung_box wraps acorr_ljungbox; stat/pvalue must match."""
        out = self.r.ljung_box(lags=1)
        ref = sm_ljungbox(self.resid, lags=[1], boxpierce=False, return_df=True)
        assert out["lb_stat"] == pytest.approx(float(ref["lb_stat"].iloc[0]), rel=1e-6)
        assert out["lb_pvalue"] == pytest.approx(float(ref["lb_pvalue"].iloc[0]), rel=1e-6)

    def test_cooks_distance_vs_statsmodels(self):
        """cooks_distance matches statsmodels OLSInfluence.cooks_distance."""
        got = self.r.cooks_distance().values
        sm_cooks = OLSInfluence(self.sm_fit).cooks_distance[0]
        assert got == pytest.approx(sm_cooks, rel=1e-6)

    def test_leverage_vs_statsmodels(self):
        """leverage matches statsmodels hat_matrix_diag."""
        got = self.r.leverage().values
        sm_hat = OLSInfluence(self.sm_fit).hat_matrix_diag
        assert got == pytest.approx(sm_hat, rel=1e-6)

    def test_dfbetas_default_matches_stata_r_convention(self):
        """Default dfbetas() matches the AUTHORITATIVE Stata/R DFBETAS.

        Ground truth is R ``stats::dfbetas`` on the committed ``df_ols`` fixture
        (tests/r/fixtures/expected/diag_estat.json), the same ground truth used
        by tests/r/tests/test_r_diagnostics.py::test_dfbetas_vector_strict_parity.
        The Stata/R leave-one-out-variance standardization is the authoritative
        target; statsmodels is NOT the reference for the default path.
        """
        import json
        from pathlib import Path

        here = Path(__file__).resolve().parent
        r_json = here.parent / "r" / "fixtures" / "expected" / "diag_estat.json"
        r_inp = here.parent / "r" / "fixtures" / "inputs" / "diag_estat_input.csv"
        R = json.loads(r_json.read_text(encoding="utf-8"))
        df = pd.read_csv(r_inp)
        r = oe.ols("y ~ x1 + x2", data=df)
        got = r.dfbetas(backend="stata_r").values
        r_mat = np.asarray(R["dfbetas"], dtype=float)
        assert got == pytest.approx(r_mat, abs=1e-6)

    def test_dfbetas_statsmodels_backend(self):
        """backend='statsmodels' reproduces statsmodels OLSInfluence.dfbetas.

        This covers the statsmodels convention through the explicit toggle
        (AGENTS.md rule 15) rather than silently comparing the default against
        statsmodels.
        """
        got = self.r.dfbetas(backend="statsmodels").values
        sm_dfb = OLSInfluence(self.sm_fit).dfbetas
        assert got == pytest.approx(sm_dfb, rel=1e-6)

    def test_dfbeta_raw_shape_and_nonnan(self):
        """Raw dfbeta() is (n, k), aligned with coefficient names, non-nan."""
        raw = self.r.dfbeta()
        assert isinstance(raw, pd.DataFrame)
        assert raw.shape == (self.X.shape[0], self.X.shape[1])
        assert list(raw.columns) == list(self.r.coefficients.index)
        assert not raw.isna().any().any()

    def test_influence_dict(self):
        """influence() bundles the expected keys with correct types."""
        inf = self.r.influence()
        assert isinstance(inf, dict)
        assert set(inf) == {
            "cooks_distance",
            "leverage",
            "dfbetas",
            "resid_studentized",
            "dffits",
        }
        assert isinstance(inf["cooks_distance"], pd.Series)
        assert isinstance(inf["leverage"], pd.Series)
        assert isinstance(inf["dfbetas"], pd.DataFrame)
        assert isinstance(inf["resid_studentized"], pd.Series)
        assert isinstance(inf["dffits"], pd.Series)
        assert len(inf["dffits"]) == self.X.shape[0]

    def test_diagnostics_table_dataframe(self):
        """diagnostics_table() returns a DataFrame."""
        table = self.r.diagnostics_table()
        assert isinstance(table, pd.DataFrame)
        assert {"test", "stat", "pvalue", "df"}.issubset(table.columns)
        assert "bgodfrey" in table["test"].values
        assert "white" in table["test"].values
        assert "ljung_box" in table["test"].values

    def test_diagnostics_still_dict(self):
        """diagnostics() still returns a dict (keeps existing test green)."""
        diag = self.r.diagnostics()
        assert isinstance(diag, dict)
        assert "jarque_bera" in diag
