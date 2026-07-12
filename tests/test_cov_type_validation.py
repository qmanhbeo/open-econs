"""Tests for the shared ``cov_type`` validation helper (core/cov_type.py).

Covers the two flagged cross-estimator inconsistencies:

1. Clear, consistent open-econs-native ``ValueError`` for any invalid
   ``cov_type`` (instead of an opaque statsmodels/linearmodels error or a
   silent fallback to a default).
2. The ``"hac"`` (any mixed case) alias for ``"HAC"`` wherever ``"HAC"`` is
   already a valid option (ols/fe/nls/driscoll_kraay), and a *clear error*
   (not silent misbehavior) everywhere else.

Every estimator that accepts ``cov_type`` is exercised: invalid raises, the
``"hac"`` alias behaves identically to ``"HAC"`` where supported, and
previously-valid values are confirmed unchanged.
"""

import numpy as np
import pandas as pd
import pytest

import open_econs as oe


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def _cross(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.0 + 2.0 * x + 0.5 * z + rng.normal(size=n)
    g = rng.integers(0, 5, size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "g": g})


def _panel(n_ent: int = 20, n_t: int = 10, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(n_ent), n_t)
    t = np.tile(np.arange(n_t), n_ent)
    x = rng.normal(size=n_ent * n_t)
    z = rng.normal(size=n_ent * n_t)
    y = 1.0 + 2.0 * x + 0.5 * z + rng.normal(size=n_ent * n_t)
    return pd.DataFrame({"y": y, "x": x, "z": z, "entity": ent, "time": t})


def _binary(n: int = 200, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.3 * x + 0.7 * z)))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _multiclass(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = rng.choice(["a", "b", "c"], size=n, p=[0.4, 0.35, 0.25])
    return pd.DataFrame({"y": y, "x": x, "z": z})


# --------------------------------------------------------------------------- #
# Shared assertions
# --------------------------------------------------------------------------- #
def _assert_clear_error(estimator_name: str, exc: Exception) -> None:
    msg = str(exc)
    assert estimator_name in msg, f"error must name the estimator: {msg!r}"
    assert "Accepted values" in msg, f"error must list accepted values: {msg!r}"


# --------------------------------------------------------------------------- #
# ols / reg
# --------------------------------------------------------------------------- #
class TestOLS:
    def test_invalid_raises_clear_error(self):
        df = _cross()
        with pytest.raises(ValueError) as e:
            oe.ols("y ~ x + z", data=df, cov_type="bogus")
        _assert_clear_error("ols()", e.value)

    def test_valid_values_unchanged(self):
        df = _cross()
        for ct in ("nonrobust", "HC0", "HC1", "HC2", "HC3"):
            r = oe.ols("y ~ x + z", data=df, cov_type=ct)
            assert r.cov_type == ct

    def test_hac_alias_equals_HAC(self):
        df = _cross()
        r_upper = oe.ols("y ~ x + z", data=df, cov_type="HAC", lags=1, time="g")
        r_lower = oe.ols("y ~ x + z", data=df, cov_type="hac", lags=1, time="g")
        r_mixed = oe.ols("y ~ x + z", data=df, cov_type="Hac", lags=1, time="g")
        assert np.allclose(r_upper.coefficients.values, r_lower.coefficients.values)
        assert np.allclose(r_upper.coefficients.values, r_mixed.coefficients.values)
        assert r_lower.cov_type == "HAC(1)"

    def test_reg_alias_same_as_ols(self):
        df = _cross()
        r = oe.reg("y ~ x + z", data=df, cov_type="HC2")
        assert r.cov_type == "HC2"


# --------------------------------------------------------------------------- #
# fe
# --------------------------------------------------------------------------- #
class TestFE:
    def test_invalid_raises_clear_error(self):
        df = _panel()
        with pytest.raises(ValueError) as e:
            oe.fe("y ~ x + z", data=df, entity="entity", time="time", cov_type="nonsense")
        _assert_clear_error("fe()", e.value)

    def test_hac_alias_equals_HAC(self):
        df = _panel()
        r_upper = oe.fe("y ~ x + z", data=df, entity="entity", time="time",
                        cov_type="HAC", lags=1)
        r_lower = oe.fe("y ~ x + z", data=df, entity="entity", time="time",
                        cov_type="hac", lags=1)
        assert np.allclose(r_upper.coefficients.values, r_lower.coefficients.values)


# --------------------------------------------------------------------------- #
# nls
# --------------------------------------------------------------------------- #
class TestNLS:
    def test_invalid_raises_clear_error(self):
        df = _cross()
        with pytest.raises(ValueError) as e:
            oe.nls("y ~ a*x + c", df, {"a": 1.0, "c": 0.0}, cov_type="garbage")
        _assert_clear_error("nls()", e.value)

    def test_valid_values_unchanged(self):
        df = _cross()
        for ct in ("nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster"):
            kw = {"cluster": "g"} if ct == "cluster" else {}
            r = oe.nls("y ~ a*x + c", df, {"a": 1.0, "c": 0.0}, cov_type=ct, **kw)
            if ct == "cluster":
                # nls relabels cluster cov_type to "cluster(<groups>)"
                assert "cluster" in r.cov_type
            else:
                assert r.cov_type == ct

    def test_hac_alias_equals_HAC(self):
        df = _cross()
        r_upper = oe.nls("y ~ a*x + c", df, {"a": 1.0, "c": 0.0},
                         cov_type="HAC", max_lags=1, time="g")
        r_lower = oe.nls("y ~ a*x + c", df, {"a": 1.0, "c": 0.0},
                         cov_type="hac", max_lags=1, time="g")
        assert np.allclose(r_upper.coefficients.values, r_lower.coefficients.values)


# --------------------------------------------------------------------------- #
# iv
# --------------------------------------------------------------------------- #
class TestIV:
    def test_invalid_raises_clear_error_not_silent_fallback(self):
        df = _cross()
        with pytest.raises(ValueError) as e:
            oe.iv("y ~ 1 | x ~ z", data=df, cov_type="bogus")
        _assert_clear_error("iv()", e.value)

    def test_valid_values_work(self):
        df = _cross()
        r_k = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="kernel")
        r_hc = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="HC2")
        assert r_k.cov_type == "kernel"
        assert r_hc.cov_type == "HC2"

    def test_hac_supported_with_lags(self):
        df = _cross()
        r = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="HAC", lags=1)
        assert r.cov_type == "HAC(1)"

    def test_hac_requires_lags(self):
        df = _cross()
        with pytest.raises(ValueError) as e:
            oe.iv("y ~ 1 | x ~ z", data=df, cov_type="HAC")
        assert "lags" in str(e.value)

    def test_hac_alias_same_as_HAC(self):
        df = _cross()
        r = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="hac", lags=2)
        assert r.cov_type == "HAC(2)"

    def test_hac_no_time_ok(self):
        df = _cross()
        r = oe.iv("y ~ 1 | x ~ z", data=df, cov_type="HAC", lags=1)
        assert r.cov_type == "HAC(1)"  # time not required


# --------------------------------------------------------------------------- #
# gmm
# --------------------------------------------------------------------------- #
class TestGMM:
    def test_invalid_raises_clear_error(self):
        df = _cross()
        with pytest.raises(ValueError) as e:
            oe.gmm("y ~ 1 | x ~ z", data=df, cov_type="weird")
        _assert_clear_error("gmm()", e.value)

    def test_valid_values_work(self):
        df = _cross()
        r_r = oe.gmm("y ~ 1 | x ~ z", data=df, cov_type="robust")
        r_c = oe.gmm("y ~ 1 | x ~ z", data=df, cov_type="cluster", cluster="g")
        assert r_r.cov_type == "robust"
        assert r_c.cov_type == "cluster"

    def test_hac_not_supported_here(self):
        df = _cross()
        with pytest.raises(ValueError) as e:
            oe.gmm("y ~ 1 | x ~ z", data=df, cov_type="HAC")
        _assert_clear_error("gmm()", e.value)


# --------------------------------------------------------------------------- #
# mlogit / logit / probit
# --------------------------------------------------------------------------- #
class TestDiscrete:
    def test_mlogit_invalid(self):
        df = _multiclass()
        with pytest.raises(ValueError) as e:
            oe.mlogit("y ~ x + z", data=df, cov_type="bad")
        _assert_clear_error("mlogit()", e.value)

    def test_mlogit_valid(self):
        df = _multiclass()
        for ct in ("nonrobust", "HC0", "HC1", "HC2", "HC3"):
            r = oe.mlogit("y ~ x + z", data=df, cov_type=ct)
            assert r.cov_type == ct

    def test_mlogit_hac_not_supported(self):
        df = _multiclass()
        with pytest.raises(ValueError) as e:
            oe.mlogit("y ~ x + z", data=df, cov_type="hac")
        _assert_clear_error("mlogit()", e.value)

    def test_logit_invalid(self):
        df = _binary()
        with pytest.raises(ValueError) as e:
            oe.logit("y ~ x + z", data=df, cov_type="bad")
        _assert_clear_error("logit()", e.value)

    def test_logit_valid(self):
        df = _binary()
        r = oe.logit("y ~ x + z", data=df, cov_type="HC2")
        assert r.cov_type == "HC2"

    def test_probit_invalid(self):
        df = _binary()
        with pytest.raises(ValueError) as e:
            oe.probit("y ~ x + z", data=df, cov_type="bad")
        _assert_clear_error("probit()", e.value)

    def test_probit_hac_not_supported(self):
        df = _binary()
        with pytest.raises(ValueError) as e:
            oe.probit("y ~ x + z", data=df, cov_type="HAC")
        _assert_clear_error("probit()", e.value)


# --------------------------------------------------------------------------- #
# did / event_study
# --------------------------------------------------------------------------- #
class TestDiD:
    def test_did_invalid_no_silent_fallback(self):
        df = _cross()
        with pytest.raises(ValueError) as e:
            oe.did("y ~ x * z", data=df, treatment="x", post="z", cov_type="junk")
        _assert_clear_error("did()", e.value)

    def test_did_robust_alias_preserved(self):
        df = _cross()
        r = oe.did("y ~ x * z", data=df, treatment="x", post="z", cov_type="robust")
        assert r.cov_type == "HC2"

    def test_did_valid_values(self):
        df = _cross()
        for ct in ("nonrobust", "HC0", "HC1", "HC2", "HC3"):
            r = oe.did("y ~ x * z", data=df, treatment="x", post="z", cov_type=ct)
            assert r.cov_type == ct

    def test_event_study_invalid(self):
        df = _cross()
        df["x_event_time"] = df["z"].astype(int)
        with pytest.raises(ValueError) as e:
            oe.event_study("y ~ x", data=df, treatment="x", post="z", cov_type="junk")
        _assert_clear_error("event_study()", e.value)

    def test_event_study_robust_alias_preserved(self):
        df = _cross()
        df["x_event_time"] = df["z"].astype(int)
        r = oe.event_study("y ~ x", data=df, treatment="x", post="z", cov_type="robust")
        assert r.cov_type == "HC2"


# --------------------------------------------------------------------------- #
# PanelContext
# --------------------------------------------------------------------------- #
class TestPanelContext:
    def _ctx(self):
        return oe.PanelContext(_panel(), entity="entity", time="time")

    def test_pooled_invalid(self):
        ctx = self._ctx()
        with pytest.raises(ValueError) as e:
            ctx.pooled("y ~ x + z", cov_type="bogus")
        _assert_clear_error("PanelContext.pooled()", e.value)

    def test_pooled_valid(self):
        ctx = self._ctx()
        # Note: pooled defaults to cluster-by-entity when the panel entity is
        # known (so the stored label may be "cluster(entity)" rather than the
        # input). We only assert here that every accepted value runs cleanly.
        for ct in ("unadjusted", "nonrobust", "HC0", "HC1", "HC2", "HC3"):
            r = ctx.pooled("y ~ x + z", cov_type=ct)
            assert r.coefficients is not None

    def test_pooled_hac_not_supported(self):
        ctx = self._ctx()
        with pytest.raises(ValueError) as e:
            ctx.pooled("y ~ x + z", cov_type="hac")
        _assert_clear_error("PanelContext.pooled()", e.value)

    def test_re_invalid(self):
        ctx = self._ctx()
        with pytest.raises(ValueError) as e:
            ctx.re("y ~ x + z", cov_type="bogus")
        _assert_clear_error("PanelContext.re()", e.value)

    def test_re_valid(self):
        ctx = self._ctx()
        for ct in ("unadjusted", "robust", "kernel", "clustered"):
            r = ctx.re("y ~ x + z", cov_type=ct)
            assert r.cov_type == ct

    def test_re_hac_not_supported(self):
        ctx = self._ctx()
        with pytest.raises(ValueError) as e:
            ctx.re("y ~ x + z", cov_type="HAC")
        _assert_clear_error("PanelContext.re()", e.value)

    def test_driscoll_kraay_invalid(self):
        ctx = self._ctx()
        with pytest.raises(ValueError) as e:
            ctx.driscoll_kraay("y ~ x + z", cov_type="robust")
        _assert_clear_error("PanelContext.driscoll_kraay()", e.value)

    def test_driscoll_kraay_hac_alias_equals_HAC(self):
        ctx = self._ctx()
        r_upper = ctx.driscoll_kraay("y ~ x + z", cov_type="HAC", lags=1)
        r_lower = ctx.driscoll_kraay("y ~ x + z", cov_type="hac", lags=1)
        r_kernel = ctx.driscoll_kraay("y ~ x + z", cov_type="kernel", lags=1)
        assert np.allclose(r_upper.coefficients.values, r_lower.coefficients.values)
        assert np.allclose(r_upper.coefficients.values, r_kernel.coefficients.values)

    def test_driscoll_kraay_kernel_still_accepted(self):
        ctx = self._ctx()
        r_hac = ctx.driscoll_kraay("y ~ x + z", cov_type="HAC", lags=2)
        r_kernel = ctx.driscoll_kraay("y ~ x + z", cov_type="kernel", lags=2)
        # "kernel" and "HAC" are aliases for the same estimator.
        assert np.allclose(r_hac.coefficients.values, r_kernel.coefficients.values)
