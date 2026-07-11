"""PanelContext API + delegation tests for the v0.6 panel-data engine.

Checks that PanelContext and the legacy Context expose the panel methods,
that every new result type renders tidy/summary/export correctly, and that
the legacy Context delegates to PanelContext transparently.
"""

import numpy as np
import pytest

import open_econs as oe


def test_panelcontext_construction(df_panel):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    assert isinstance(repr(pc), str)
    assert "entity" in repr(pc)


def test_panelcontext_all_estimators_run(df_panel):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    fe = pc.fe("y ~ x + z")
    re = pc.re("y ~ x + z")
    pooled = pc.pooled("y ~ x + z")
    fd = pc.diff("y ~ x + z")
    dk = pc.driscoll_kraay("y ~ x + z")
    h = pc.hausman(fe, re)
    # Pooled OLS / DK / FE / RE keep all observations; first-difference drops
    # the first period of each entity, so its nobs is strictly smaller.
    for r in (fe, re, pooled, dk):
        assert r.nobs == len(df_panel)
    assert fd.nobs < len(df_panel)
    assert h.statistic >= 0.0


def test_random_effects_result_api(df_panel):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    re = pc.re("y ~ x + z")
    t = re.tidy()
    assert list(t.columns) == ["Variable", "Coef", "Std Err", "z", "P>|z|", "0.025", "0.975"]
    assert "x" in t["Variable"].values
    assert isinstance(re.summary(), str)
    assert re.n_entities > 0 and re.n_time > 0
    assert re.sigma2_effects >= 0.0
    assert re.sigma2_eps >= 0.0


def test_hausman_result_api(df_panel):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    fe = pc.fe("y ~ x + z")
    re = pc.re("y ~ x + z")
    h = pc.hausman(fe, re)
    t = h.tidy()
    assert list(t.columns) == ["term", "value"]
    assert isinstance(h.summary(), str)
    assert h.df == 2
    assert h.rejected_at(0.05) in (True, False)
    d = h.to_dict()
    assert "statistic" in d and "rejected_at_default_alpha" in d


def test_first_difference_result_api(df_panel):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    fd = pc.diff("y ~ x + z")
    assert fd.method == "first-difference"
    assert isinstance(fd, oe.core.results.OLSResult)
    assert isinstance(fd.summary(), str)
    assert fd.vcov().shape == (2, 2)


def test_random_effects_vcov_and_export(df_panel, tmp_path):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    re = pc.re("y ~ x + z")
    v = re.vcov()
    assert v.shape == (3, 3)
    assert np.all(np.isfinite(v.values))
    csv = tmp_path / "re.csv"
    re.export(str(csv))
    assert csv.exists()
    jsonp = tmp_path / "re.json"
    re.export(str(jsonp))
    assert jsonp.exists()


def test_context_delegates_pooled_and_hausman(df_panel):
    c = oe.Context(df_panel)
    pooled = c.pooled("y ~ x + z")
    assert pooled.nobs == len(df_panel)
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    fe = pc.fe("y ~ x + z")
    re = pc.re("y ~ x + z")
    h = c.hausman(fe, re)
    assert h.statistic >= 0.0


def test_context_diff_and_dk_require_entity_time(df_panel):
    c = oe.Context(df_panel)
    with pytest.raises(Exception):
        c.diff("y ~ x + z")
    with pytest.raises(Exception):
        c.driscoll_kraay("y ~ x + z")


def test_panelcontext_cross_section_delegation(df_panel):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    r = pc.ols("y ~ x + z")
    assert r.nobs == len(df_panel)


def test_panelcontext_gmm_defaults_to_entity_cluster(df_panel):
    # ctx.gmm(formula) with no explicit cluster must be identical to calling the
    # top-level gmm() directly with cluster=<entity col> and cov_type="cluster".
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    formula = "y ~ x | z"
    delegated = pc.gmm(formula)
    direct = oe.gmm(formula, data=df_panel, cluster="entity", cov_type="cluster")
    assert delegated.cov_type == "cluster"
    assert delegated.step == "two-step"
    assert np.allclose(delegated.coefficients.values, direct.coefficients.values)
    assert np.allclose(delegated.std_errors.values, direct.std_errors.values)
    assert np.allclose(delegated.hansen_j, direct.hansen_j)


def test_panelcontext_gmm_explicit_cluster_and_covtype_win(df_panel):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    formula = "y ~ x | z"
    # explicit cluster= overrides the entity default (and cov_type still
    # defaults to "cluster").
    r_cluster = pc.gmm(formula, cluster="time")
    direct_cluster = oe.gmm(
        formula, data=df_panel, cluster="time", cov_type="cluster"
    )
    assert r_cluster.cov_type == "cluster"
    assert np.allclose(r_cluster.std_errors.values, direct_cluster.std_errors.values)
    # explicit cov_type="robust" is honored and no entity cluster is injected.
    r_robust = pc.gmm(formula, cov_type="robust")
    direct_robust = oe.gmm(formula, data=df_panel, cov_type="robust")
    assert r_robust.cov_type == "robust"
    assert np.allclose(r_robust.std_errors.values, direct_robust.std_errors.values)
