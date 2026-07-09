from datetime import datetime
from typing import Any

import pandas as pd

from open_econs._version import __version__
from open_econs.core.panel_results import (
    FirstDifferenceResult,
    HausmanResult,
    RandomEffectsResult,
    _hausman_test,
    _panel_ols_result,
    _re_result_from_fit,
)
from open_econs.core.results import OLSResult


class PanelContext:
    """A context that remembers panel structure (entity and time) for repeated estimation.

    Once created with ``entity`` and ``time`` columns, the panel methods
    (``pooled``, ``fe``, ``re``, ``diff``, ``driscoll_kraay``, ``hausman``)
    no longer need those columns re-specified on every call.

    Parameters
    ----------
    data : pd.DataFrame
        The working dataset.
    entity : str, optional
        Column name for the panel entity (unit) index.
    time : str, optional
        Column name for the panel time index.

    Examples
    --------
    >>> import open_econs as oe
    >>> ctx = oe.PanelContext(df, entity="country", time="year")
    >>> fe = ctx.fe("y ~ x1 + x2")
    >>> re = ctx.re("y ~ x1 + x2")
    >>> h = ctx.hausman(fe, re)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        self._data = data
        self._entity = entity
        self._time = time

    # ── internals ──────────────────────────────────────────────────

    def _panel_index(self, require: bool = True) -> pd.DataFrame:
        """Return the data with a (entity, time) MultiIndex set."""
        if self._entity is None or self._time is None:
            if require:
                raise ValueError(
                    "PanelContext requires entity= and time= to be set (at "
                    "construction or per-call) for this method."
                )
            return self._data
        pdf = self._data.set_index([self._entity, self._time])
        if pdf.index.duplicated().any():
            raise ValueError(
                "Panel data contains duplicate (entity, time) pairs, which make "
                "the within / first-difference transforms unidentified. Drop or "
                "aggregate duplicates before estimating."
            )
        return pdf

    @staticmethod
    def _ensure_intercept(formula: str) -> str:
        """Return *formula* with an explicit intercept unless one is suppressed."""
        lhs, rhs = formula.split("~", 1)
        rhs = rhs.strip()
        if rhs.startswith("0") or "0 +" in rhs or "~0" in formula:
            return formula
        if rhs.startswith("1") or "1 +" in rhs:
            return formula
        return f"{lhs.strip()} ~ 1 + {rhs}"

    def _rhs(self, formula: str) -> str:
        return formula.split("~", 1)[1].strip()

    def _capture(self, **kwargs: Any) -> dict[str, Any]:
        kwargs["timestamp"] = str(datetime.now())
        kwargs["package_version"] = __version__
        return kwargs

    # ── estimators ─────────────────────────────────────────────────

    def pooled(
        self,
        formula: str,
        cov_type: str = "unadjusted",
    ) -> Any:
        """Pooled OLS (constant coefficients across entities and time)."""
        from open_econs.models.linear.ols import ols as _ols

        # linearmodels uses "unadjusted"; statsmodels uses "nonrobust".
        sm_cov = "nonrobust" if cov_type == "unadjusted" else cov_type
        return _ols(formula=formula, data=self._data, cov_type=sm_cov)

    def driscoll_kraay(self, formula: str) -> Any:
        """Pooled OLS with Driscoll-Kraay (spatial/time-series-robust) SEs."""
        from linearmodels.panel import PooledOLS

        pdf = self._panel_index(require=True)
        re_formula = self._ensure_intercept(formula)
        call = self._capture(
            formula=formula, method="driscoll_kraay",
            entity=self._entity, time=self._time,
        )
        fit = PooledOLS.from_formula(re_formula, pdf).fit(cov_type="kernel")
        return _panel_ols_result(
            OLSResult, formula, self._rhs(re_formula), fit, "driscoll-kraay", call,
        )

    def fe(
        self,
        formula: str,
        cov_type: str = "HC1",
        cluster: str | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> Any:
        """Fixed-effects (within) estimator using absorbed entity/time dummies."""
        from open_econs.models.linear.fe import fe as _fe

        ent = entity if entity is not None else self._entity
        tm = time if time is not None else self._time
        if ent is None and tm is None:
            raise ValueError(
                "fe() requires entity/time either in PanelContext or as arguments."
            )
        return _fe(
            formula=formula, data=self._data, entity=ent, time=tm,
            cov_type=cov_type, cluster=cluster,
        )

    def re(
        self,
        formula: str,
        cov_type: str = "unadjusted",
    ) -> RandomEffectsResult:
        """Random-effects (GLS) estimator with Swamy-Arora variance components."""
        from linearmodels.panel import RandomEffects

        pdf = self._panel_index(require=True)
        re_formula = self._ensure_intercept(formula)
        call = self._capture(
            formula=formula, method="random_effects",
            entity=self._entity, time=self._time, cov_type=cov_type,
        )
        fit = RandomEffects.from_formula(re_formula, pdf).fit(cov_type=cov_type)
        return _re_result_from_fit(formula, self._rhs(re_formula), fit, cov_type, call)

    def diff(self, formula: str) -> FirstDifferenceResult:
        """First-difference estimator (removes time-invariant unobserved effects)."""
        from linearmodels.panel import FirstDifferenceOLS

        pdf = self._panel_index(require=True)
        call = self._capture(
            formula=formula, method="first_difference",
            entity=self._entity, time=self._time,
        )
        fit = FirstDifferenceOLS.from_formula(formula, pdf).fit()
        return _panel_ols_result(
            FirstDifferenceResult, formula, self._rhs(formula), fit,
            "nonrobust", call,
        )

    def hausman(
        self,
        fe_result: Any,
        re_result: Any,
        alpha: float = 0.05,
    ) -> HausmanResult:
        """Hausman test of FE vs RE consistency."""
        return _hausman_test(fe_result, re_result, alpha=alpha)

    # ── delegation to the cross-sectional estimators ───────────────

    def ols(self, formula: str, cluster: str | None = None, cov_type: str = "HC2") -> Any:
        from open_econs.models.linear.ols import ols as _ols

        return _ols(formula=formula, data=self._data, cluster=cluster, cov_type=cov_type)

    def logit(self, formula: str, cov_type: str = "nonrobust") -> Any:
        from open_econs.models.discrete.logit import logit as _logit

        return _logit(formula=formula, data=self._data, cov_type=cov_type)

    def probit(self, formula: str, cov_type: str = "nonrobust") -> Any:
        from open_econs.models.discrete.probit import probit as _probit

        return _probit(formula=formula, data=self._data, cov_type=cov_type)

    def did(self, formula: str, treatment: str, post: str, cluster: str | None = None, cov_type: str = "HC2") -> Any:
        from open_econs.models.causal.did import did as _did

        return _did(formula=formula, data=self._data, treatment=treatment, post=post,
                    cluster=cluster, cov_type=cov_type)

    def event_study(self, formula: str, treatment: str, post: str, cluster: str | None = None,
                    cov_type: str = "HC2", omitted_period: int = -1) -> Any:
        from open_econs.models.causal.did import event_study as _event_study

        return _event_study(formula=formula, data=self._data, treatment=treatment, post=post,
                            cluster=cluster, cov_type=cov_type, omitted_period=omitted_period)

    def balance(self, treatment: str, covariates: list[str] | None = None) -> pd.DataFrame:
        from open_econs.models.causal.balance import balance as _balance

        return _balance(self._data, treatment=treatment, covariates=covariates)

    def __repr__(self) -> str:
        head = f"PanelContext ({self._data.shape[0]} rows, {self._data.shape[1]} cols)"
        if self._entity is not None and self._time is not None:
            head += f" [entity={self._entity!r}, time={self._time!r}]"
        return head
