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


_UNSET = object()  # sentinel to distinguish "not passed" from explicit None


class PanelContext:
    """A context that remembers panel structure (entity and time) for repeated estimation.

    Once created with ``entity`` and ``time`` columns, the panel methods
    (``pooled``, ``fe``, ``re``, ``diff``, ``driscoll_kraay``, ``hausman``)
    no longer need those columns re-specified on every call.  Cross-sectional
    estimators that do not need panel structure are also exposed as thin
    delegates (``ols``, ``logit``, ``probit``, ``did``, ``event_study``,
    ``balance``, ``gmm``) that forward the context's data and, where relevant,
    default ``cluster`` to the context's entity column.

    Panel HAC standard errors are exposed under two alias names that mean the
    same thing: ``"kernel"`` (the historical linearmodels spelling, used by
    :meth:`driscoll_kraay`) and ``"HAC"`` (this project's preferred spelling,
    matching ``ols``/``fe``/``nls``).  Both invoke the same period-aggregation
    Bartlett-kernel Newey-West estimator and produce identical results; see
    :meth:`driscoll_kraay` for the details and the equivalence proof in
    ``tests/test_panelcontext_hac.py``.

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
        cluster: str | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> Any:
        """Pooled OLS (constant coefficients across entities and time).

        When the panel ``entity`` is known (either from the context or via the
        ``entity`` argument) standard errors default to panel-robust
        cluster-by-entity inference, matching Stata's ``xtreg, vce(cluster)``
        for a pooled specification.  Pass ``cluster`` to override, or
        ``cov_type="nonrobust"`` with ``cluster=None`` for iid errors.
        """
        from open_econs.models.linear.ols import ols as _ols

        ent = entity if entity is not None else self._entity
        # linearmodels uses "unadjusted"; statsmodels uses "nonrobust".
        sm_cov = "nonrobust" if cov_type == "unadjusted" else cov_type
        # Pooled OLS defaults to panel-robust (cluster-by-entity) inference when
        # the panel entity is known, matching Stata's xtreg, vce(cluster) for a
        # pooled model.  Pass cov_type="nonrobust" (with cluster=None) for iid
        # errors, or an explicit cluster column to override.
        if cluster is not None:
            use_cluster = cluster
        elif ent is not None and cov_type != "nonrobust":
            use_cluster = ent
        else:
            use_cluster = None
        return _ols(
            formula=formula, data=self._data, cov_type=sm_cov, cluster=use_cluster,
        )

    def driscoll_kraay(
        self,
        formula: str,
        cov_type: str = "kernel",
        lags: int | None = None,
        kernel: str | None = None,
    ) -> Any:
        """Pooled OLS with Driscoll-Kraay (panel HAC / Newey-West) SEs.

        ``cov_type`` accepts ``"kernel"`` (the historical linearmodels name) or
        ``"HAC"`` (this project's preferred name, matching ``ols``/``fe``/``nls``).
        Both spellings invoke the same linearmodels Driscoll-Kraay estimator and
        therefore produce identical results. ``"HAC"`` is the recommended spelling
        going forward; ``"kernel"`` is retained for backward compatibility.

        The Driscoll-Kraay estimator is a period-aggregation (Arellano /
        Driscoll-Kraay) Bartlett-kernel Newey-West long-run variance: the score
        contributions ``x_it * e_it`` are summed *within* each time period across
        entities, then a Bartlett-kernel HAC is applied *across* periods. This is
        the same convention implemented by
        :func:`open_econs.core.cov.newey_west_cov` (verified to agree to machine
        precision; see ``tests/test_panelcontext_hac.py``), so ``"HAC"`` here is
        genuinely the same computation as the project's own panel HAC -- not a
        second, divergent implementation.

        Parameters
        ----------
        cov_type : {"kernel", "HAC"}, default "kernel"
            Which name to use for the panel-HAC estimator. Both are aliases.
        lags : int, optional
            HAC bandwidth (maximum lag). Maps to linearmodels' ``bandwidth``.
            If omitted, linearmodels' rule-of-thumb ``floor(4 * (T / 100) ** (2 / 9))``
            is used, where ``T`` is the number of time periods. (Note: the
            project's own :func:`newey_west_cov` requires an explicit ``max_lags``
            and does not apply this default.)
        kernel : str, optional
            Kernel name forwarded to linearmodels -- ``"newey-west"``/``"bartlett"``,
            ``"quadratic-spectral"``/``"qs"``/``"andrews"``, or ``"parzen"``/``"gallant"``.
            Defaults to the Bartlett / Newey-West kernel.
        """
        from linearmodels.panel import PooledOLS

        if cov_type not in ("kernel", "HAC"):
            raise ValueError(
                f"driscoll_kraay cov_type must be 'kernel' or 'HAC' (got {cov_type!r}). "
                "Both name the same period-aggregation Newey-West estimator; 'HAC' is preferred."
            )
        pdf = self._panel_index(require=True)
        re_formula = self._ensure_intercept(formula)
        call = self._capture(
            formula=formula, method="driscoll_kraay",
            entity=self._entity, time=self._time,
            cov_type=cov_type, lags=lags, kernel=kernel,
        )
        cov_config: dict[str, Any] = {"cov_type": "kernel"}
        if lags is not None:
            cov_config["bandwidth"] = int(lags)
        if kernel is not None:
            cov_config["kernel"] = kernel
        fit = PooledOLS.from_formula(re_formula, pdf).fit(**cov_config)
        return _panel_ols_result(
            OLSResult, formula, self._rhs(re_formula), fit, "driscoll-kraay", call,
        )

    def fe(
        self,
        formula: str,
        cov_type: str = "HC1",
        cluster: str | None = None,
        entity: str | object = _UNSET,
        time: str | object | None = _UNSET,
    ) -> Any:
        """Fixed-effects (within) estimator using absorbed entity/time dummies."""
        from open_econs.models.linear.fe import fe as _fe
        from typing import cast

        ent = cast("str | None", self._entity if entity is _UNSET else entity)
        tm = cast("str | None", self._time if time is _UNSET else time)
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

    def abond(
        self,
        formula: str,
        lags: int = 1,
        max_iv_lag: int | None = None,
        step: str = "two-step",
        exogenous: list[str] | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> Any:
        """Arellano-Bond dynamic panel estimator (difference GMM)."""
        from open_econs.models.linear.abond import abond as _abond

        ent = entity if entity is not None else self._entity
        tm = time if time is not None else self._time
        if ent is None or tm is None:
            raise ValueError(
                "abond() requires entity/time either in PanelContext or as arguments."
            )
        return _abond(
            formula=formula, data=self._data, entity=ent, time=tm,
            lags=lags, max_iv_lag=max_iv_lag, step=step, exogenous=exogenous,
        )

    # ── delegation to the cross-sectional estimators ───────────────

    def ols(self, formula: str, cluster: str | None = None, cov_type: str = "HC2") -> Any:
        from open_econs.models.linear.ols import ols as _ols

        return _ols(formula=formula, data=self._data, cluster=cluster, cov_type=cov_type)

    def gmm(
        self,
        formula: str,
        *,
        step: str = "two-step",
        cov_type: str | None = None,
        cluster: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Linear GMM (cross-sectional delegate).

        Thin wrapper around the top-level :func:`gmm` that forwards the
        context's data (``self._data``).  When the caller does not supply
        them, ``cluster`` defaults to the context's entity column and
        ``cov_type`` defaults to ``"cluster"`` -- mirroring the default-context
        injection convention of :meth:`pooled`; explicit ``cluster=`` /
        ``cov_type=`` always win and are forwarded unchanged.  The top-level
        estimator rejects a cluster with a non-cluster ``cov_type``, so no
        entity cluster is injected in that case.

        For dynamic panel models with lagged dependent variables and
        instrument construction from the panel structure, see :meth:`abond`;
        this method provides plain linear GMM with entity-clustering
        convenience only.
        """
        from open_econs.models.linear.gmm import gmm as _gmm

        use_cov = "cluster" if cov_type is None else cov_type
        # Mirror PanelContext.pooled(): inject the entity cluster only when the
        # covariance type is (or defaults to) clustering.
        use_cluster = self._entity if (use_cov == "cluster" and cluster is None) else cluster
        return _gmm(
            formula=formula, data=self._data, step=step,
            cov_type=use_cov, cluster=use_cluster, **kwargs,
        )

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
