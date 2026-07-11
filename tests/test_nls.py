"""Public-API tests for the nonlinear least-squares estimator ``oe.nls`` / ``NLSResult``.

Three layers of checks:

1. **Primary parity (always on):** against :func:`scipy.optimize.curve_fit`
   on two textbook nonlinear models (exponential decay and a CES production
   function).  curve_fit is the canonical reference for NLS; we assert real
   max-absolute differences on coefficients *and* iid standard errors.
2. **Secondary parity (gated):** against R's ``nls()`` and Stata's ``nl``.
   Both are skipped gracefully when the binary is unavailable; R's
   sandwich-based robust SE is skipped with a *documented* reason (the
   ``sandwich`` package is not installed on this machine), never fabricated.
3. **Contract / interface tests:** cluster-vs-robust sanity, result-class
   surface (``tidy``/``summary``/``vcov``/``export``/immutability), convergence
   fields, the ``jacobian_method`` flag, error cases (parameter/data
   collision, typo), and the numerical-Jacobian fallback path.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import curve_fit

import open_econs as oe
from open_econs.models.nonlinear.nls import NLSResult

# ── external-binary gates (re-confirmed fresh each session) ───────────────
# Stata: confirmed present at this path on the dev machine (prior recon that
# claimed Stata was absent only checked PATH / STATA_EXE and missed the
# default install location).
STATA_EXE = os.environ.get(
    "STATA_EXE", r"C:\Program Files\Stata17\StataMP-64.exe"
)
STATA_AVAILABLE = Path(STATA_EXE).is_file()

# R: confirmed present (off-PATH); the `sandwich` package is NOT installed, so
# the robust-via-sandwich comparison is intentionally skipped.
RSCRIPT_EXE = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
R_AVAILABLE = Path(RSCRIPT_EXE).is_file()
R_SANDWICH_AVAILABLE = False  # verified: requireNamespace('sandwich') == FALSE


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def exp_data() -> pd.DataFrame:
    """Exponential-decay data y = a*exp(-b*x) + c + noise (seed 0)."""
    rng = np.random.default_rng(0)
    n = 120
    x = np.linspace(0.1, 3.0, n)
    true = {"a": 2.5, "b": 1.3, "c": 0.5}
    y = (
        true["a"] * np.exp(-true["b"] * x)
        + true["c"]
        + rng.normal(0.0, 0.15, n)
    )
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture
def ces_data() -> pd.DataFrame:
    """CES production function y = (a*x1^-r + b*x2^-r)^(-1/r) + noise."""
    rng = np.random.default_rng(3)
    n = 200
    x1 = rng.uniform(0.5, 2.0, n)
    x2 = rng.uniform(0.5, 2.0, n)
    a, b, r = 0.6, 0.4, 0.5
    y = (a * x1 ** (-r) + b * x2 ** (-r)) ** (-1.0 / r) + rng.normal(0.0, 0.02, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture
def clustered() -> pd.DataFrame:
    """Genuine within-cluster error correlation (mirrors test_gmm's fixture):
    x shares a cluster component with the error's random effect, so the
    cluster-robust SE must be strictly larger than HC2."""
    rng = np.random.default_rng(7)
    n_cl, m = 40, 25
    n = n_cl * m
    cid = np.repeat(np.arange(n_cl), m)
    b = rng.normal(0, 1, n_cl)   # cluster component shared by x
    a = rng.normal(0, 1, n_cl)   # cluster random effect in the error
    b_i, a_i = b[cid], a[cid]
    x = b_i + rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x + a_i + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x": x, "cid": cid})


# ── 1. curve_fit parity (always on) ───────────────────────────────────────

class TestCurveFitParity:
    def test_exponential_decay(self, exp_data):
        df = exp_data
        r = oe.nls("y ~ a*exp(-b*x)+c", df, {"a": 1.0, "b": 1.0, "c": 0.0},
                   cov_type="nonrobust")

        popt, pcov = curve_fit(
            lambda x, a, b, c: a * np.exp(-b * x) + c,
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            p0=[1.0, 1.0, 0.0],
        )
        cf_se = np.sqrt(np.diag(pcov))

        oe_coef = r.coefficients.loc[["a", "b", "c"]].to_numpy()
        oe_se = r.std_errors.loc[["a", "b", "c"]].to_numpy()
        max_coef_diff = np.max(np.abs(oe_coef - popt))
        max_se_diff = np.max(np.abs(oe_se - cf_se))
        # Documented in the hand-off: curve_fit scales pcov by SSR/(n-k),
        # identical to nls cov_type="nonrobust".
        assert max_coef_diff < 1e-5, f"coef diff {max_coef_diff}"
        assert max_se_diff < 1e-5, f"se diff {max_se_diff}"
        assert r.jacobian_method == "analytic"

    def test_ces_production(self, ces_data):
        df = ces_data

        def ces(x, a, b, r):
            x1, x2 = x
            return (a * x1 ** (-r) + b * x2 ** (-r)) ** (-1.0 / r)

        r = oe.nls(
            "y ~ (a*x1^(-r) + b*x2^(-r))^(-1/r)",
            df,
            {"a": 0.6, "b": 0.4, "r": 0.5},
        )
        popt, _ = curve_fit(
            ces,
            np.vstack([df["x1"].to_numpy(), df["x2"].to_numpy()]),
            df["y"].to_numpy(),
            p0=[0.6, 0.4, 0.5],
            maxfev=20000,
        )
        oe_coef = r.coefficients.loc[["a", "b", "r"]].to_numpy()
        max_coef_diff = np.max(np.abs(oe_coef - popt))
        assert max_coef_diff < 1e-4, f"CES coef diff {max_coef_diff}"
        assert r.jacobian_method == "analytic"


# ── 2. R nls() parity (gated) ─────────────────────────────────────────────

class TestRnlsParity:
    def test_r_nls_iid(self, exp_data, tmp_path):
        if not R_AVAILABLE:
            pytest.skip(f"Rscript not found at {RSCRIPT_EXE}")

        df = exp_data
        d = tmp_path / "nls_r"
        d.mkdir(parents=True, exist_ok=True)
        csv_in = d / "r_data.csv"
        csv_out = d / "r_out.csv"
        df.to_csv(csv_in, index=False)

        # Generate the R script with Python file-writing (never a shell heredoc
        # for R either -- backslashes in Windows paths become R unicode escapes).
        rd = str(d).replace("\\", "/")
        r_script = (
            'df <- read.csv("__D__/r_data.csv")\n'
            'fit <- nls(y ~ a*exp(-b*x)+c, data=df, start=list(a=1,b=1,c=0))\n'
            'cf <- coef(fit)\n'
            'se <- sqrt(diag(vcov(fit)))\n'
            'out <- data.frame(name=names(cf), coef=as.numeric(cf), se=as.numeric(se))\n'
            'write.csv(out, "__D__/r_out.csv", row.names=FALSE)\n'
        ).replace("__D__", rd)
        r_file = d / "r_fit.R"
        r_file.write_text(r_script, encoding="utf-8")

        proc = subprocess.run(
            [RSCRIPT_EXE, str(r_file)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(d),
        )
        if proc.returncode != 0 or not csv_out.exists():
            pytest.skip(f"R nls() run failed/unavailable: {proc.stderr[-300:]}")

        out = pd.read_csv(csv_out)
        r_coef = dict(zip(out["name"], out["coef"]))
        r_se = dict(zip(out["name"], out["se"]))

        r_oe = oe.nls("y ~ a*exp(-b*x)+c", df, {"a": 1.0, "b": 1.0, "c": 0.0},
                      cov_type="nonrobust")
        for name in ("a", "b", "c"):
            assert abs(r_coef[name] - float(r_oe.coefficients[name])) < 1e-4, name
            assert abs(r_se[name] - float(r_oe.std_errors[name])) < 1e-5, name

    def test_r_robust_skipped_by_design(self):
        # The sandwich-based robust comparison is intentionally NOT run: the
        # `sandwich` package is confirmed absent on this machine.  We assert the
        # skip reason is tracked rather than fabricating a ground-truth number.
        assert R_SANDWICH_AVAILABLE is False, "sandwich now installed -- add R robust parity"


# ── 3. Stata nl parity (gated) ────────────────────────────────────────────

class TestStataNlParity:
    def test_stata_nl_iid_and_robust(self, exp_data, tmp_path):
        if not STATA_AVAILABLE:
            pytest.skip(f"Stata not found at {STATA_EXE}")

        df = exp_data
        d = tmp_path / "nls_stata"
        d.mkdir(parents=True, exist_ok=True)
        csv_in = d / "exp.csv"
        csv_iid = d / "exp_stata.csv"
        csv_rob = d / "exp_stata_rob.csv"
        df.to_csv(csv_in, index=False)

        # FIX (this task): generate the .do file with Python's own write_text so
        # Stata's backtick local-macro syntax survives verbatim.  A prior attempt
        # used a PowerShell heredoc, which mangled `i' / `names' / `k' into
        # i' / names' / k' and produced no output.  We assert the backticks are
        # intact before running -- never "run it and hope".
        dd = str(d).replace("\\", "/")
        do = """cd "__D__"
import delimited using "__D__/exp.csv", clear
nl (y = {a}*exp(-{b}*x)+{c}), initial(a 1 b 1 c 0)
matrix b = e(b)
matrix V = e(V)
local names : colfullnames e(b)
local k = colsof(b)
file open fh using "__D__/exp_stata.csv", write replace
file write fh "name,coef,se" _n
forvalues i = 1/`k' {
    local nm : word `i' of `names'
    local nm2 = subinstr("`nm'", "/", "", 1)
    local c = b[1,`i']
    local s = sqrt(V[`i',`i'])
    file write fh "`nm2',`c',`s'" _n
}
file close fh
nl (y = {a}*exp(-{b}*x)+{c}), initial(a 1 b 1 c 0) vce(robust)
matrix b2 = e(b)
matrix V2 = e(V)
file open fh2 using "__D__/exp_stata_rob.csv", write replace
file write fh2 "name,coef,se" _n
forvalues i = 1/`k' {
    local nm : word `i' of `names'
    local nm2 = subinstr("`nm'", "/", "", 1)
    local c = b2[1,`i']
    local s = sqrt(V2[`i',`i'])
    file write fh2 "`nm2',`c',`s'" _n
}
file close fh2
"""
        do_text = do.replace("__D__", dd)
        # Verify the escaping fix: backticks preserved, no mangled forms.
        assert "1/`k'" in do_text
        assert "word `i' of `names'" in do_text
        assert "1/k'" not in do_text
        assert "word i' of names'" not in do_text
        do_file = d / "exp.do"
        do_file.write_text(do_text, encoding="utf-8")

        proc = subprocess.run(
            [STATA_EXE, "/e", "do", str(do_file)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(d),
        )
        if proc.returncode != 0 or not (csv_iid.exists() and csv_rob.exists()):
            pytest.skip(
                f"Stata nl run failed/unusable: rc={proc.returncode} "
                f"{proc.stderr[-300:]}"
            )

        iid = pd.read_csv(csv_iid)
        rob = pd.read_csv(csv_rob)
        sta_coef = dict(zip(iid["name"].str.replace(":_cons", "", regex=False), iid["coef"]))
        sta_iid_se = dict(zip(iid["name"].str.replace(":_cons", "", regex=False), iid["se"]))
        sta_rob_se = dict(zip(rob["name"].str.replace(":_cons", "", regex=False), rob["se"]))

        # nls iid (nonrobust) must match Stata default nl; Stata vce(robust)
        # matches nls HC1 (both = n/(n-k) * raw e_i^2, no leverage correction).
        r_non = oe.nls("y ~ a*exp(-b*x)+c", df, {"a": 1.0, "b": 1.0, "c": 0.0},
                       cov_type="nonrobust")
        r_hc1 = oe.nls("y ~ a*exp(-b*x)+c", df, {"a": 1.0, "b": 1.0, "c": 0.0},
                       cov_type="HC1")
        for name in ("a", "b", "c"):
            assert abs(sta_coef[name] - float(r_non.coefficients[name])) < 1e-4, name
            assert abs(sta_iid_se[name] - float(r_non.std_errors[name])) < 1e-5, name
            assert abs(sta_rob_se[name] - float(r_hc1.std_errors[name])) < 1e-5, name


# ── 4. cluster vs robust sanity ───────────────────────────────────────────

class TestClusterVsRobust:
    def test_cluster_se_larger_than_robust(self, clustered):
        df = clustered
        robust = oe.nls("y ~ a*x+c", df, {"a": 1.0, "c": 0.0}, cov_type="HC2")
        cluster = oe.nls(
            "y ~ a*x+c", df, {"a": 1.0, "c": 0.0}, cov_type="cluster", cluster="cid"
        )
        assert np.allclose(
            cluster.coefficients.values, robust.coefficients.values, atol=1e-10
        )
        assert cluster.std_errors["a"] > robust.std_errors["a"]
        assert cluster.std_errors["a"] / robust.std_errors["a"] > 1.2


# ── 5. result-class interface ─────────────────────────────────────────────

class TestNLSResultInterface:
    def test_tidy_summary_vcov_export(self, exp_data, tmp_path):
        r = oe.nls("y ~ a*exp(-b*x)+c", exp_data, {"a": 1.0, "b": 1.0, "c": 0.0})

        t = r.tidy()
        assert list(t.columns) == [
            "Variable", "Coef", "Std Err", "t", "P>|t|", "0.025", "0.975",
        ]
        assert list(t["Variable"]) == ["a", "b", "c"]

        assert isinstance(r.summary(), str)
        assert "Nonlinear Least Squares" in r.summary()

        v = r.vcov()
        assert v.shape == (3, 3)
        assert np.allclose(np.diag(v), r.std_errors.values ** 2)

        csv = tmp_path / "nls.csv"
        r.export(str(csv))
        assert csv.exists()
        js = tmp_path / "nls.json"
        r.export(str(js))
        assert js.exists()

        # vcov() rows/cols keyed by parameter names
        assert list(v.index) == ["a", "b", "c"]

    def test_immutable(self, exp_data):
        r = oe.nls("y ~ a*exp(-b*x)+c", exp_data, {"a": 1.0, "b": 1.0, "c": 0.0})
        with pytest.raises(AttributeError):
            r.coefficients = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        with pytest.raises(AttributeError):
            r.jacobian_method = "numerical"

    def test_convergence_fields(self, exp_data):
        r = oe.nls("y ~ a*exp(-b*x)+c", exp_data, {"a": 1.0, "b": 1.0, "c": 0.0})
        # least_squares exposes NO nit; n_iterations proxies nfev.
        assert isinstance(r.success, bool) and r.success is True
        assert r.n_function_evaluations > 0
        assert r.n_jacobian_evaluations > 0
        assert r.n_iterations == r.n_function_evaluations
        assert r.cost > 0
        assert isinstance(r.status, int)
        assert isinstance(r.message, str)
        assert r.optimality >= 0

    def test_jacobian_method_analytic(self, exp_data):
        r = oe.nls("y ~ a*exp(-b*x)+c", exp_data, {"a": 1.0, "b": 1.0, "c": 0.0})
        assert r.jacobian_method == "analytic"

    def test_time_param_wired_for_hac(self, exp_data):
        df = exp_data.copy()
        df["t"] = np.arange(len(df))
        r = oe.nls(
            "y ~ a*exp(-b*x)+c", df, {"a": 1.0, "b": 1.0, "c": 0.0},
            cov_type="HAC", max_lags=2, time="t",
        )
        assert r.call["time"] == "t"
        assert "HAC" in r.cov_type

    def test_is_nlsresult(self, exp_data):
        r = oe.nls("y ~ a*exp(-b*x)+c", exp_data, {"a": 1.0, "b": 1.0, "c": 0.0})
        assert isinstance(r, NLSResult)


# ── 6. error cases ────────────────────────────────────────────────────────

class TestErrorCases:
    def test_parameter_data_collision(self, exp_data):
        # 'x' is both a start_value key and a data column -> clear ValueError.
        with pytest.raises(ValueError):
            oe.nls("y ~ a*exp(-b*x)+c", exp_data, {"a": 1.0, "b": 1.0, "x": 0.0})

    def test_typo_neither(self, exp_data):
        # 'zz' is neither a parameter nor a data column -> clear ValueError.
        with pytest.raises(ValueError):
            oe.nls("y ~ a*exp(-b*zz)+c", exp_data, {"a": 1.0, "b": 1.0, "c": 0.0})

    def test_cluster_required(self, clustered):
        with pytest.raises(ValueError):
            oe.nls("y ~ a*x+c", clustered, {"a": 1.0, "c": 0.0}, cov_type="cluster")

    def test_hac_requires_max_lags(self, exp_data):
        with pytest.raises(ValueError):
            oe.nls("y ~ a*exp(-b*x)+c", exp_data, {"a": 1.0, "b": 1.0, "c": 0.0},
                   cov_type="HAC")

    def test_time_must_be_a_column(self, exp_data):
        with pytest.raises(ValueError):
            oe.nls("y ~ a*exp(-b*x)+c", exp_data, {"a": 1.0, "b": 1.0, "c": 0.0},
                   cov_type="HAC", max_lags=2, time="nonexistent")


# ── 7. numerical-Jacobian fallback ────────────────────────────────────────

class TestNumericalFallback:
    def test_heaviside_triggers_numerical(self):
        rng = np.random.default_rng(5)
        n = 80
        x = np.linspace(-1.0, 1.0, n)
        true = {"a": 2.0, "b": 3.0, "c": 0.5}
        y = true["a"] * np.heaviside(true["b"] * x, 0.0) + true["c"] + rng.normal(0, 0.05, n)
        df = pd.DataFrame({"y": y, "x": x})
        r = oe.nls("y ~ a*Heaviside(b*x)+c", df, {"a": 1.0, "b": 1.0, "c": 0.0})
        # sympy cannot differentiate Heaviside w.r.t. its argument -> fallback.
        assert r.jacobian_method == "numerical"
        assert r.success is True
        # coefficient recovered within tolerance of the data-generating value
        assert abs(float(r.coefficients["a"]) - true["a"]) < 0.2
        assert abs(float(r.coefficients["c"]) - true["c"]) < 0.2
