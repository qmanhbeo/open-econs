"""PanelContext "kernel" / "HAC" alias tests.

Investigation (see hand-off): linearmodels' ``cov_type="kernel"`` maps to its
``DriscollKraay`` covariance estimator, which is a period-aggregation
(Arellano / Driscoll-Kraay) Bartlett-kernel Newey-West long-run variance -- the
same convention as :func:`open_econs.core.cov.newey_west_cov`. ``PanelContext``
therefore accepts ``"HAC"`` as an alias for ``"kernel"`` in :meth:`driscoll_kraay`
(``"kernel"`` retained for backward compatibility); they must produce identical
output, and both must agree with the project's own ``newey_west_cov``.
"""

import numpy as np
import pandas as pd
import pytest

import open_econs as oe
from open_econs.core.cov import _as_int_labels, newey_west_cov


def _make_panel(n: int = 15, t: int = 10, seed: int = 42) -> pd.DataFrame:
    # Regressor names x/z are arbitrary; the _ensure_intercept substring bug
    # (matching "1 +" inside x1-style names) is fixed in panel_context.py.
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n), t)
    times = np.tile(np.arange(t), n)
    x = rng.standard_normal(n * t)
    z = rng.standard_normal(n * t)
    # AR(1)-ish heteroskedastic-within-period errors so HAC actually matters.
    e = rng.standard_normal(n * t) * (0.3 + 0.5 * np.sin(times))
    y = 1.5 + 0.8 * x - 1.2 * z + e
    return pd.DataFrame({"y": y, "x": x, "z": z, "id": ids, "t": times})


def test_kernel_and_hac_aliases_identical():
    df = _make_panel()
    ctx = oe.PanelContext(df, entity="id", time="t")
    r_kernel = ctx.driscoll_kraay("y ~ x + z", cov_type="kernel")
    r_hac = ctx.driscoll_kraay("y ~ x + z", cov_type="HAC")
    # The two cov_type strings must be exact aliases (same estimator).
    assert np.allclose(r_kernel.vcov().values, r_hac.vcov().values, atol=1e-12)
    assert np.allclose(r_kernel.std_errors.values, r_hac.std_errors.values, atol=1e-12)
    assert np.allclose(r_kernel.coefficients.values, r_hac.coefficients.values, atol=1e-12)


def test_invalid_cov_type_rejected():
    df = _make_panel()
    ctx = oe.PanelContext(df, entity="id", time="t")
    with pytest.raises(ValueError):
        ctx.driscoll_kraay("y ~ x + z", cov_type="robust")
    with pytest.raises(ValueError):
        ctx.driscoll_kraay("y ~ x + z", cov_type="HC1")


def test_hac_lags_maps_to_bandwidth():
    df = _make_panel()
    ctx = oe.PanelContext(df, entity="id", time="t")
    # Explicit lags=3 must equal passing bandwidth=3 to the "kernel" name.
    r_hac3 = ctx.driscoll_kraay("y ~ x + z", cov_type="HAC", lags=3)
    r_kern3 = ctx.driscoll_kraay("y ~ x + z", cov_type="kernel", lags=3)
    assert np.allclose(r_hac3.vcov().values, r_kern3.vcov().values, atol=1e-12)
    # And it must differ from the default-bandwidth run (T=10 -> rule-of-thumb=2).
    r_def = ctx.driscoll_kraay("y ~ x + z", cov_type="HAC")
    assert not np.allclose(r_hac3.vcov().values, r_def.vcov().values, atol=1e-10)


def test_hac_matches_project_newey_west_cov():
    # linearmodels Driscoll-Kraay (its default applies an N/(N-K) debiasing,
    # equivalent to newey_west_cov(adjust=True)) is the same period-aggregation
    # Bartlett-kernel Newey-West computation as the project's own cov routine.
    df = _make_panel()
    ctx = oe.PanelContext(df, entity="id", time="t")
    r = ctx.driscoll_kraay("y ~ x + z", cov_type="HAC", lags=2)

    X = np.column_stack([np.ones(df.shape[0]), df["x"].values, df["z"].values])
    resid = df["y"].values - X @ r.coefficients.values
    time_labels = _as_int_labels(df["t"].values)

    own = newey_west_cov(X, resid, max_lags=2, cluster=time_labels, adjust=True)
    assert np.allclose(r.vcov().values, own, atol=1e-6)
