"""Stata + R parity tests for the multinomial logit estimator (mlogit).

The baseline category is pinned explicitly on every backend (Stata
``baseoutcome(1)``, open-econs ``base=1``, R ``factor(y, levels=1:3)``) because
Stata's default base is the *most frequent* category while statsmodels/R use the
*first (sorted)* category — they disagree by default.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from .stata_runner import FIXTURES_DIR, read_stata

S = read_stata("mlogit_basic")


def _suffix(v: str) -> str:
    return "cons" if v == "Intercept" else v


def _load_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_DIR / "df_mlogit.csv")


def _rscript_exe() -> str | None:
    p = os.environ.get("R_SCRIPT") or shutil.which("Rscript")
    if p and os.path.isfile(p):
        return p
    cand = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
    return cand if os.path.isfile(cand) else None


@pytest.mark.stata
class TestMlogitCoefficients:
    @pytest.fixture(autouse=True)
    def _run(self):
        self.s = S
        self.oe = oe.mlogit("y ~ x1 + x2", data=_load_df(), base=1)

    def test_nobs(self):
        assert self.oe.nobs == int(self.s["N"])

    def test_coefficients_outcome2(self):
        b = self.oe.coefficients.loc[2]
        npt.assert_allclose(
            [b["Intercept"], b["x1"], b["x2"]],
            [self.s["b_2_cons"], self.s["b_2_x1"], self.s["b_2_x2"]],
            rtol=1e-6,
        )

    def test_coefficients_outcome3(self):
        b = self.oe.coefficients.loc[3]
        npt.assert_allclose(
            [b["Intercept"], b["x1"], b["x2"]],
            [self.s["b_3_cons"], self.s["b_3_x1"], self.s["b_3_x2"]],
            rtol=1e-6,
        )

    def test_tidy_has_all_outcomes(self):
        outcomes = set(self.oe.tidy()["Outcome"])
        assert outcomes == {2, 3}


@pytest.mark.stata
class TestMlogitMargins:
    """Average marginal effects must match Stata's delta-method AMEs (rtol 1e-6)."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.s = S
        self.oe = oe.mlogit("y ~ x1 + x2", data=_load_df(), base=1)
        self.me = self.oe.margins()

    def test_margins_include_base(self):
        assert set(self.me.keys()) == {1, 2, 3}

    @pytest.mark.parametrize("cat", [1, 2, 3])
    def test_margins_ame(self, cat):
        dfm = self.me[cat].set_index("Variable")
        sx1 = self.s[f"me_{cat}_x1"]
        sx2 = self.s[f"me_{cat}_x2"]
        npt.assert_allclose([dfm.loc["x1", "dy/dx"], dfm.loc["x2", "dy/dx"]],
                            [sx1, sx2], rtol=1e-6, atol=1e-8)

    def test_base_ame_identity(self):
        """baseline AME == -sum(non-baseline AMEs); AMEs sum to zero per regressor."""
        base = self.me[self.oe.base_category].set_index("Variable")["dy/dx"]
        nonbase = sum(self.me[c].set_index("Variable")["dy/dx"]
                      for c in self.oe.non_base_categories)
        resid = (base + nonbase).abs().max()
        assert resid < 1e-8, f"baseline AME identity residual too large: {resid}"


@pytest.mark.stata
class TestMlogitSEs:
    """Robust (HC1) and cluster SEs vs Stata; cluster must exceed robust (real
    within-cluster correlation)."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.s = S
        self.df = _load_df()
        self.oe_rob = oe.mlogit("y ~ x1 + x2", data=self.df, base=1, cov_type="HC1")
        self.oe_clu = oe.mlogit("y ~ x1 + x2", data=self.df, base=1, cluster="cluster")

    def test_robust_se_vs_stata(self):
        maxd = 0.0
        for cat in [2, 3]:
            se = self.oe_rob.std_errors.loc[cat]
            for v in ["Intercept", "x1", "x2"]:
                d = abs(se[v] - self.s[f"se_rob_{cat}_{_suffix(v)}"])
                maxd = max(maxd, d)
        # Stata vce(robust) == HC1; tiny dof discrepancy -> 1e-3 tolerance.
        assert maxd < 1e-3, f"robust SE max diff {maxd}"
        print(f"[mlogit] robust SE max abs diff vs Stata: {maxd:.2e}")

    def test_cluster_se_vs_stata(self):
        maxd = 0.0
        for cat in [2, 3]:
            se = self.oe_clu.std_errors.loc[cat]
            for v in ["Intercept", "x1", "x2"]:
                d = abs(se[v] - self.s[f"se_clu_{cat}_{_suffix(v)}"])
                maxd = max(maxd, d)
        # Stata vce(cluster) vs statsmodels "cluster":
        # Stata uses a G/(G-1) degrees-of-freedom adjustment on the cluster-robust
        # covariance (G = number of clusters); statsmodels uses a different (n-1
        # style) dof. This is a *known, documented* discrepancy, NOT an unresolved
        # bug: the two estimators target the same point estimates but differ only
        # in the cluster dof scaling. Empirically the per-SE absolute difference
        # lands in the ~1e-4-1e-3 range on this fixture, so the tolerance below is
        # intentional. Do not "fix" the tolerance tighter without first replicating
        # Stata's exact G/(G-1) cluster dof correction.
        assert maxd < 1e-3, f"cluster SE max diff {maxd}"
        print(f"[mlogit] cluster SE max abs diff vs Stata: {maxd:.2e}")

    def test_cluster_exceeds_robust(self):
        ratios = []
        for cat in [2, 3]:
            for v in ["Intercept", "x1", "x2"]:
                ratios.append(self.oe_clu.std_errors.loc[cat, v]
                              / self.oe_rob.std_errors.loc[cat, v])
        ratios = np.array(ratios)
        assert ratios.min() > 1.0, "cluster SE must be >= robust SE"
        assert ratios.max() >= 1.3, f"cluster effect too weak (max ratio {ratios.max():.2f})"
        print(f"[mlogit] cluster/robust SE ratios min={ratios.min():.3f} max={ratios.max():.3f}")


class TestMlogitPredict:
    def test_predict_shape_and_probabilities(self):
        r = oe.mlogit("y ~ x1 + x2", data=_load_df(), base=1)
        pr = r.predict()
        assert isinstance(pr, pd.DataFrame)
        assert pr.shape == (len(_load_df()), 3)
        assert list(pr.columns) == ["1", "2", "3"]
        npt.assert_allclose(pr.sum(axis=1).to_numpy(), 1.0, rtol=1e-9)


@pytest.mark.r
class TestMlogitR:
    """R nnet::multinom cross-check (coefficients only). Skipped if R is absent."""

    @pytest.fixture(autouse=True)
    def _run(self):
        self.df = _load_df()
        self.oe = oe.mlogit("y ~ x1 + x2", data=self.df, base=1)

    def test_r_coefficients(self):
        rscript = _rscript_exe()
        if rscript is None:
            pytest.skip("Rscript not available")
        with tempfile.TemporaryDirectory() as td:
            out_csv = os.path.join(td, "r_coef.csv").replace("\\", "/")
            r_script = os.path.join(td, "mlogit_r.R")
            in_csv = str(FIXTURES_DIR / "df_mlogit.csv").replace("\\", "/")
            code = (
                'suppressMessages(library(nnet))\n'
                f'd <- read.csv("{in_csv}")\n'
                'd$y <- factor(d$y, levels = c(1, 2, 3))\n'
                'fit <- multinom(y ~ x1 + x2, data = d, trace = FALSE, maxit = 500)\n'
                f'write.csv(as.matrix(coef(fit)), "{out_csv}", row.names = TRUE)\n'
            )
            with open(r_script, "w") as fh:
                fh.write(code)
            res = subprocess.run([rscript, r_script], capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Rscript failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
            cf = pd.read_csv(out_csv, index_col=0)
            cf.index = cf.index.astype(str)
        maxd = 0.0
        for cat in [2, 3]:
            b = self.oe.coefficients.loc[cat]
            for v in ["Intercept", "x1", "x2"]:
                rv = "(Intercept)" if v == "Intercept" else v
                d = abs(b[v] - cf.loc[str(cat), rv])
                maxd = max(maxd, d)
        # R nnet uses a different optimizer; coefficients match to ~1e-4.
        assert maxd < 1e-3, f"R coefficient max diff {maxd}"
        print(f"[mlogit] R nnet::multinom coefficient max abs diff: {maxd:.2e}")
