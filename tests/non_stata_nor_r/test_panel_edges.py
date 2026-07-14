"""Edge-case / adversarial tests for the v0.6 panel-data engine."""

import numpy as np
import pandas as pd
import pytest

import open_econs as oe


def test_fe_without_entity_or_time_raises(df_panel):
    with pytest.raises(ValueError):
        oe.fe("y ~ x + z", data=df_panel)


def test_panelcontext_re_requires_entity_time(df_panel):
    pc = oe.PanelContext(df_panel)  # no entity/time
    with pytest.raises(ValueError):
        pc.re("y ~ x + z")


def test_panelcontext_fe_without_entity_time_raises(df_panel):
    pc = oe.PanelContext(df_panel)
    with pytest.raises(ValueError):
        pc.fe("y ~ x + z")


def test_hausman_no_common_terms_raises(df_panel):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    fe_x = pc.fe("y ~ x")
    fe_z = pc.fe("y ~ z")
    # Coefficients are disjoint ('x' vs 'z') -> no common terms to compare.
    with pytest.raises(ValueError):
        pc.hausman(fe_x, fe_z)


def test_duplicate_panel_index_raises(df_panel_dup_index):
    pc = oe.PanelContext(df_panel_dup_index, entity="entity", time="time")
    with pytest.raises(ValueError):
        pc.re("y ~ x + z")


def test_fe_single_entity_works(df_panel_single_entity):
    r = oe.fe("y ~ x + z", data=df_panel_single_entity, entity="entity")
    assert np.isfinite(r.coefficients).all()
    assert r.nobs == len(df_panel_single_entity)


def test_fe_single_time_period_works():
    np.random.seed(2)
    n = 40
    entity = np.arange(n)
    time = np.zeros(n, dtype=int)
    x = np.random.normal(0, 1, n)
    y = 1.2 * x + np.random.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x": x, "entity": entity, "time": time})
    r = oe.fe("y ~ x", data=df, time="time")
    assert np.isfinite(r.coefficients["x"])


def test_constant_outcome_does_not_crash(df_panel):
    df = df_panel.copy()
    df["y"] = 5.0
    r = oe.fe("y ~ x + z", data=df, entity="entity", time="time")
    assert np.isfinite(r.coefficients).all()
    assert np.isfinite(r.std_errors).all()


def test_hausman_singular_difference_is_safe():
    # No entity fixed effect -> FE and RE are numerically identical, so
    # V_fe - V_re is ~0; the test must return a finite statistic, not crash.
    np.random.seed(4)
    n_unit, n_time = 30, 5
    n = n_unit * n_time
    entity = np.repeat(np.arange(n_unit), n_time)
    time = np.tile(np.arange(n_time), n_unit)
    x = np.random.normal(0, 1, n)
    z = np.random.normal(0, 1, n)
    y = 1.5 * x - 0.7 * z + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x": x, "z": z, "entity": entity, "time": time})
    pc = oe.PanelContext(df, entity="entity", time="time")
    fe = pc.fe("y ~ x + z")
    re = pc.re("y ~ x + z")
    h = pc.hausman(fe, re)
    assert np.isfinite(h.statistic)
    assert 0.0 <= h.p_value <= 1.0


def test_panelcontext_repr_shows_structure(df_panel):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    assert "entity" in repr(pc) and "time" in repr(pc)


def test_context_gains_panel_methods(df_panel):
    c = oe.Context(df_panel)
    for meth in ("fe", "re", "pooled", "diff", "driscoll_kraay", "hausman"):
        assert hasattr(c, meth), meth


def test_context_fe_matches_panelcontext(df_panel):
    c = oe.Context(df_panel)
    fe_ctx = c.fe("y ~ x + z", entity="entity", time="time")
    fe_pc = oe.PanelContext(df_panel, entity="entity", time="time").fe("y ~ x + z")
    assert np.allclose(fe_ctx.coefficients.values, fe_pc.coefficients.values)


def test_fd_vcov_works(df_panel):
    fd = oe.PanelContext(df_panel, entity="entity", time="time").diff("y ~ x + z")
    v = fd.vcov()
    assert v.shape == (2, 2)
    assert np.all(np.isfinite(v.values))


def test_re_predict_newdata(df_panel):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    re = pc.re("y ~ x + z")
    pred = re.predict(newdata=df_panel)
    assert len(pred) == len(df_panel)
    assert np.all(np.isfinite(pred.values))


def test_re_export_roundtrip(df_panel, tmp_path):
    pc = oe.PanelContext(df_panel, entity="entity", time="time")
    re = pc.re("y ~ x + z")
    p = tmp_path / "re.json"
    re.export(str(p))
    assert p.exists()
    d = re.to_dict()
    assert "results" in d
