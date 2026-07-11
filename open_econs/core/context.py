from typing import Any

import pandas as pd


class Context:
    """A context that remembers a dataset for repeated estimation.

    Once created, every estimator method forwards to the top-level
    function with ``data=self.data`` injected automatically.

    Parameters
    ----------
    data : pd.DataFrame
        The working dataset.

    Examples
    --------
    >>> import open_econs as oe
    >>> ctx = oe.Context(df)
    >>> r1 = ctx.ols("income ~ education + age")
    >>> r2 = ctx.oaxaca("income ~ education + age + female", by="female")
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data

    def ols(
        self,
        formula: str,
        cluster: str | None = None,
        cov_type: str = "HC2",
    ) -> Any:
        from open_econs.models.linear.ols import ols as _ols

        return _ols(formula=formula, data=self._data, cluster=cluster, cov_type=cov_type)

    def oaxaca(
        self,
        formula: str,
        by: str,
        decomposition_type: str = "two-fold",
    ) -> Any:
        from open_econs.models.decomposition.oaxaca import oaxaca as _oaxaca

        return _oaxaca(
            formula=formula,
            data=self._data,
            by=by,
            decomposition_type=decomposition_type,
        )

    def logit(self, formula: str, cov_type: str = "nonrobust") -> Any:
        from open_econs.models.discrete.logit import logit as _logit

        return _logit(formula=formula, data=self._data, cov_type=cov_type)

    def probit(self, formula: str, cov_type: str = "nonrobust") -> Any:
        from open_econs.models.discrete.probit import probit as _probit

        return _probit(formula=formula, data=self._data, cov_type=cov_type)

    def nls(
        self,
        formula: str,
        start_values: dict[str, float],
        *,
        cov_type: str = "HC2",
        cluster: str | list[str] | None = None,
        max_lags: int | None = None,
        time: str | None = None,
        **solver_kwargs: Any,
    ) -> Any:
        from open_econs.models.nonlinear.nls import nls as _nls

        return _nls(
            formula=formula,
            data=self._data,
            start_values=start_values,
            cov_type=cov_type,
            cluster=cluster,
            max_lags=max_lags,
            time=time,
            **solver_kwargs,
        )

    def vif(self, formula: str) -> pd.Series:
        if "~" in formula:
            rhs = formula.split("~", 1)[1].strip()
        else:
            rhs = formula.strip()

        from formulaic import Formula
        matrices = Formula(rhs).get_model_matrix(self._data, na_action="drop")
        X = matrices.rhs if hasattr(matrices, "rhs") else matrices
        X_arr = X.values.astype(float)

        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_values = []
        for i in range(X_arr.shape[1]):
            vif_values.append(variance_inflation_factor(X_arr, i))

        result = pd.Series(vif_values, index=X.columns, name="VIF")
        result.index.name = "Variable"
        return result

    def did(
        self,
        formula: str,
        treatment: str,
        post: str,
        cluster: str | None = None,
        cov_type: str = "HC2",
    ) -> Any:
        from open_econs.models.causal.did import did as _did

        return _did(
            formula=formula,
            data=self._data,
            treatment=treatment,
            post=post,
            cluster=cluster,
            cov_type=cov_type,
        )

    def event_study(
        self,
        formula: str,
        treatment: str,
        post: str,
        cluster: str | None = None,
        cov_type: str = "HC2",
        omitted_period: int = -1,
    ) -> Any:
        from open_econs.models.causal.did import event_study as _event_study

        return _event_study(
            formula=formula,
            data=self._data,
            treatment=treatment,
            post=post,
            cluster=cluster,
            cov_type=cov_type,
            omitted_period=omitted_period,
        )

    def balance(
        self,
        treatment: str,
        covariates: list[str] | None = None,
        weights: str | None = None,
    ) -> pd.DataFrame:
        from open_econs.models.causal.balance import balance as _balance

        return _balance(
            self._data,
            treatment=treatment,
            covariates=covariates,
            weights=weights,
        )

    # ── panel-data engine (delegates to a transient PanelContext) ──

    def _to_panel(
        self, entity: str | None = None, time: str | None = None,
    ) -> Any:
        from open_econs.core.panel_context import PanelContext

        return PanelContext(self._data, entity=entity, time=time)

    def pooled(
        self,
        formula: str,
        cov_type: str = "unadjusted",
        cluster: str | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> Any:
        return self._to_panel(entity, time).pooled(
            formula, cov_type=cov_type, cluster=cluster, entity=entity, time=time,
        )

    def fe(
        self,
        formula: str,
        entity: str | None = None,
        time: str | None = None,
        cov_type: str = "HC1",
        cluster: str | None = None,
    ) -> Any:
        return self._to_panel(entity, time).fe(
            formula, cov_type=cov_type, cluster=cluster,
        )

    def re(
        self,
        formula: str,
        entity: str,
        time: str,
        cov_type: str = "unadjusted",
    ) -> Any:
        return self._to_panel(entity, time).re(formula, cov_type=cov_type)

    def diff(self, formula: str, entity: str, time: str) -> Any:
        return self._to_panel(entity, time).diff(formula)

    def driscoll_kraay(self, formula: str, entity: str, time: str) -> Any:
        return self._to_panel(entity, time).driscoll_kraay(formula)

    def hausman(self, fe_result: Any, re_result: Any, alpha: float = 0.05) -> Any:
        return self._to_panel().hausman(fe_result, re_result, alpha=alpha)

    def abond(
        self,
        formula: str,
        entity: str,
        time: str,
        lags: int = 1,
        max_iv_lag: int | None = None,
        step: str = "two-step",
        exogenous: list[str] | None = None,
    ) -> Any:
        return self._to_panel(entity, time).abond(
            formula, lags=lags, max_iv_lag=max_iv_lag, step=step, exogenous=exogenous,
        )

    def __repr__(self) -> str:
        return f"Context ({self._data.shape[0]} rows, {self._data.shape[1]} cols)"