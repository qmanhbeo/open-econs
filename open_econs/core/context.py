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
        cov_type: str = "HC1",
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

    def __repr__(self) -> str:
        return f"Context(data: {self._data.shape[0]} rows, {self._data.shape[1]} cols)"