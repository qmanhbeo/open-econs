"""Regression tests for PanelContext intercept detection (``_ensure_intercept``).

The historical implementation matched the raw substring ``"1 +"`` against the
RHS of a formula. A regressor name ending in a digit before ``" + "`` -- e.g.
``"y ~ x1 + x2"`` contains the literal substring ``"1 +"`` (from ``x1 + x2``)
-- wrongly convinced the helper that an intercept was already present, so it
silently failed to add one. linearmodels' ``from_formula`` does *not* auto-add
an intercept on this path, so the dropped intercept shifted every other
coefficient. The fix tokenizes the RHS on ``+``/``-`` and matches ``1``/``0``
only as standalone terms (see ``panel_context.py``).
"""

import numpy as np
import pandas as pd

import open_econs as oe


def _make_panel(n: int = 15, t: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n), t)
    times = np.tile(np.arange(t), n)
    x1 = rng.standard_normal(n * t)
    x2 = rng.standard_normal(n * t)
    # Known intercept (1.5) so we can also check recovery, not just presence.
    e = rng.standard_normal(n * t) * (0.3 + 0.5 * np.sin(times))
    y = 1.5 + 0.8 * x1 - 1.2 * x2 + e
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "id": ids, "t": times})


def _has_intercept(result) -> bool:
    return any("Intercept" in str(name) for name in result.coefficients.index)


def test_driscoll_kraay_adds_intercept_for_digit_regressors():
    df = _make_panel()
    ctx = oe.PanelContext(df, entity="id", time="t")
    # Buggy behavior produced coefficients ['x1', 'x2'] with no intercept.
    r = ctx.driscoll_kraay("y ~ x1 + x2")
    assert _has_intercept(r), "intercept silently dropped for y ~ x1 + x2"
    assert list(r.coefficients.index) == ["Intercept", "x1", "x2"]
    # Adding the intercept explicitly must yield the same estimates.
    r_explicit = ctx.driscoll_kraay("y ~ 1 + x1 + x2")
    assert np.allclose(r.coefficients.values, r_explicit.coefficients.values, atol=1e-10)
    # And the known intercept is recovered (regressors are not mean-zero).
    assert abs(r.coefficients["Intercept"] - 1.5) < 0.2


def test_re_adds_intercept_for_digit_regressors():
    df = _make_panel()
    ctx = oe.PanelContext(df, entity="id", time="t")
    r = ctx.re("y ~ x1 + x2")
    assert _has_intercept(r), "intercept silently dropped for re(y ~ x1 + x2)"
    assert list(r.coefficients.index) == ["Intercept", "x1", "x2"]


def test_explicit_intercept_not_duplicated():
    df = _make_panel()
    ctx = oe.PanelContext(df, entity="id", time="t")
    r = ctx.driscoll_kraay("y ~ 1 + x1 + x2")
    names = [str(n) for n in r.coefficients.index]
    assert names.count("Intercept") == 1
    # trailing "+ 1" (previously also mis-handled) must not double the intercept.
    r_tail = ctx.driscoll_kraay("y ~ x1 + x2 + 1")
    assert names == [str(n) for n in r_tail.coefficients.index]


def test_explicit_intercept_suppression_preserved():
    df = _make_panel()
    ctx = oe.PanelContext(df, entity="id", time="t")
    for formula in ("y ~ 0 + x1 + x2", "y ~ x1 + x2 - 1", "y ~ x1 + x2 + 0"):
        r = ctx.driscoll_kraay(formula)
        assert not _has_intercept(r), f"intercept should be suppressed for {formula!r}"
        assert list(r.coefficients.index) == ["x1", "x2"]


def test_latent_bug_no_longer_needs_x_z_workaround():
    # Names like year2020 / lag1 at the operator boundary must no longer fool
    # the detector.
    df = _make_panel()
    df = df.rename(columns={"x1": "year2020", "x2": "lag1"})
    ctx = oe.PanelContext(df, entity="id", time="t")
    r = ctx.driscoll_kraay("y ~ year2020 + lag1")
    assert list(r.coefficients.index) == ["Intercept", "year2020", "lag1"]
