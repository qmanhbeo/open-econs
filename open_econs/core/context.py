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

    def __repr__(self) -> str:
        return f"Context(data: {self._data.shape[0]} rows, {self._data.shape[1]} cols)"