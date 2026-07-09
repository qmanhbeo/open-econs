import matplotlib

matplotlib.use("Agg")  # never open a window in tests

import numpy as np
import pandas as pd
import pytest

import open_econs as oe


@pytest.fixture(scope="module")
def ols_data():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.5 * x - 0.7 * z + rng.normal(size=n)
    treat = (rng.uniform(size=n) > 0.5).astype(int)
    post = (rng.uniform(size=n) > 0.5).astype(int)
    return pd.DataFrame({"y": y, "x": x, "z": z, "treat": treat, "post": post})


def test_ols_export_formats(ols_data):
    r = oe.ols("y ~ x + z", data=ols_data)
    latex = r.to_latex(caption="OLS", label="tab:ols")
    html = r.to_html(caption="OLS")
    assert "\\begin{tabular}" in latex
    assert "OLS" in latex
    assert "<table" in html
    d = r.to_dict()
    assert d["formula"] == "y ~ x + z"
    assert "results" in d and len(d["results"]) == 3


def test_latex_html_run_on_all_result_types():
    rng = np.random.default_rng(3)
    n = 60
    T = 5
    ent = np.repeat(np.arange(n), T)
    t = np.tile(np.arange(T), n)
    x = rng.normal(size=n * T)
    y = 0.5 * x + rng.normal(size=n * T)
    mu = rng.normal(size=n)
    y_level = y + mu[ent]
    df = pd.DataFrame({"y": y_level, "x": x, "firm": ent, "year": t})

    z = x + rng.normal(size=n * T)
    df["z"] = z
    ols_r = oe.ols("y ~ x", data=df)
    iv_r = oe.iv("y ~ x | z", data=df)

    pc = oe.PanelContext(df, entity="firm", time="year")
    fe_r = pc.fe("y ~ x")
    re_r = pc.re("y ~ x")
    diff_r = pc.diff("y ~ x")

    df["b"] = (rng.uniform(size=n * T) > 0.5).astype(int)
    logit_r = oe.logit("b ~ x", data=df)
    probit_r = oe.probit("b ~ x", data=df)

    did_r = oe.did(
        "y ~ treat * post",
        data=df.assign(treat=(df["firm"] % 2).astype(int), post=(df["year"] > 2).astype(int)),
        treatment="treat",
        post="post",
    )
    bal_r = oe.balance(df, treatment="b")
    ox_r = oe.oaxaca("y ~ x + b", data=df, by="b")
    ab_r = oe.abond("y ~ x", data=df, entity="firm", time="year")

    for r in [ols_r, iv_r, fe_r, re_r, diff_r, logit_r, probit_r, did_r, ox_r, ab_r]:
        latex = r.to_latex()
        html = r.to_html()
        assert isinstance(latex, str) and "\\begin{tabular}" in latex
        assert isinstance(html, str) and "<table" in html

    assert isinstance(bal_r, pd.DataFrame)
