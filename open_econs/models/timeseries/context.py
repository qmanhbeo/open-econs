from __future__ import annotations

import numpy as np
from typing import Any

import pandas as pd

from open_econs.models.timeseries.arima import arma, arima
from open_econs.models.timeseries.garch import garch
from open_econs.models.timeseries.unitroot import (
    adf,
    dfgls,
    kpss,
    pp,
    zivot_andrews,
)
from open_econs.models.timeseries.ardl import (
    ardl_fit,
    ardl_select_order,
    uecm_fit,
)
from open_econs.models.timeseries.var import (
    johansen_cointegration,
    var_fit,
    var_select_order,
    vecm_fit,
)


class TimeSeriesContext:
    """A context that remembers time-series structure for repeated estimation.

    This is the time-series analogue of :class:`open_econs.PanelContext`.  Just
    as ``tsset`` in Stata records the time index, frequency and lag-operator
    conventions for a dataset, a ``TimeSeriesContext`` stores the ``time``
    column, the panel/observation frequency, and keeps the working ``data`` so
    that subsequent calls to :meth:`garch`, :meth:`adf`, :meth:`pp`, :meth:`arima`,
    etc. do not need to re-specify the series each time.

    Parameters
    ----------
    data : pd.DataFrame
        The working dataset (one column is the series of interest).
    time : str, optional
        Column name for the time index.  When set, it defines the ordering used
        by estimators and is the ``tsset`` equivalent.
    freq : str, optional
        Pandas-style frequency string (e.g. ``"D"``, ``"M"``, ``"Q"``, ``"Y"``)
        recording the sampling frequency / lag-operator convention.  Informational
        only -- it is stored and surfaced but not currently enforced by the
        wrapped backends.

    Examples
    --------
    >>> import open_econs as oe
    >>> ctx = oe.TimeSeriesContext(df, time="date")
    >>> g = ctx.garch("y", p=1, q=1)
    >>> a = ctx.adf("y")
    >>> m = ctx.arima("y", order=(1, 0, 1))
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time: str | None = None,
        freq: str | None = None,
    ) -> None:
        self._data = data
        self._time = time
        self._freq = freq

    # ── internals ──────────────────────────────────────────────────

    def _series(self, y: str | pd.Series | np.ndarray) -> pd.Series:
        """Resolve a column name / Series / ndarray to a 1-D Series.

        If a column name is given and a ``time`` index is set, the series is
        returned in time order (the ``tsset`` ordering).
        """
        import numpy as np

        if isinstance(y, str):
            if y not in self._data.columns:
                raise KeyError(f"Column {y!r} not found in the context data.")
            s = self._data[y].astype(float)
        elif isinstance(y, pd.Series):
            s = y.astype(float).reset_index(drop=True)
        elif isinstance(y, np.ndarray):
            s = pd.Series(y).astype(float)
        else:
            raise TypeError("y must be a column name, Series, or ndarray")

        if isinstance(y, str) and self._time is not None and self._time in self._data.columns:
            s = self._data.set_index(self._time)[y].astype(float)
            s = s.reset_index(drop=True)
        return s

    def _capture(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("time", self._time)
        kwargs.setdefault("freq", self._freq)
        return kwargs

    # ── estimators ─────────────────────────────────────────────────

    def garch(self, y: str | pd.Series | np.ndarray, **kwargs: Any) -> Any:
        """Fit a GARCH-family volatility model on a series.  See :func:`open_econs.garch`."""
        return garch(self._series(y), **kwargs)

    def adf(self, y: str | pd.Series | np.ndarray, **kwargs: Any) -> Any:
        """Augmented Dickey-Fuller test.  See :func:`open_econs.adf`."""
        return adf(self._series(y), **kwargs)

    def pp(self, y: str | pd.Series | np.ndarray, **kwargs: Any) -> Any:
        """Phillips-Perron test.  See :func:`open_econs.pp`."""
        return pp(self._series(y), **kwargs)

    def kpss(self, y: str | pd.Series | np.ndarray, **kwargs: Any) -> Any:
        """KPSS stationarity test.  See :func:`open_econs.kpss`."""
        return kpss(self._series(y), **kwargs)

    def dfgls(self, y: str | pd.Series | np.ndarray, **kwargs: Any) -> Any:
        """DF-GLS (ERS) unit-root test.  See :func:`open_econs.dfgls`."""
        return dfgls(self._series(y), **kwargs)

    def zivot_andrews(self, y: str | pd.Series | np.ndarray, **kwargs: Any) -> Any:
        """Zivot-Andrews structural-break unit-root test.  See :func:`open_econs.zivot_andrews`."""
        return zivot_andrews(self._series(y), **kwargs)

    def arima(
        self, y: str | pd.Series | np.ndarray, **kwargs: Any
    ) -> Any:
        """ARIMA model.  See :func:`open_econs.arima`."""
        return arima(self._series(y), **kwargs)

    def arma(self, y: str | pd.Series | np.ndarray, **kwargs: Any) -> Any:
        """ARMA model.  See :func:`open_econs.arma`."""
        return arma(self._series(y), **kwargs)

    def var_fit(self, columns: list[str] | pd.DataFrame | np.ndarray, **kwargs: Any) -> Any:
        """Fit a VAR model.  See :func:`open_econs.var_fit`."""
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, list):
            data = self._data[columns].astype(float)
        else:
            data = columns
        return var_fit(data, **kwargs)

    def var_select_order(self, columns: list[str] | pd.DataFrame | np.ndarray, **kwargs: Any) -> Any:
        """Select VAR lag order.  See :func:`open_econs.var_select_order`."""
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, list):
            data = self._data[columns].astype(float)
        else:
            data = columns
        return var_select_order(data, **kwargs)

    def johansen_cointegration(self, columns: list[str] | pd.DataFrame | np.ndarray, **kwargs: Any) -> Any:
        """Johansen cointegration test.  See :func:`open_econs.johansen_cointegration`."""
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, list):
            data = self._data[columns].astype(float)
        else:
            data = columns
        return johansen_cointegration(data, **kwargs)

    def vecm_fit(self, columns: list[str] | pd.DataFrame | np.ndarray, **kwargs: Any) -> Any:
        """Fit a VECM.  See :func:`open_econs.vecm_fit`."""
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, list):
            data = self._data[columns].astype(float)
        else:
            data = columns
        return vecm_fit(data, **kwargs)

    def ardl_fit(self, y: str, **kwargs: Any) -> Any:
        """Fit an ARDL(p, q1, ..., qk) model.  See :func:`open_econs.ardl_fit`."""
        return ardl_fit(self._data, y, **kwargs)

    def uecm_fit(self, y: str, **kwargs: Any) -> Any:
        """Fit the UECM (error-correction) form of an ARDL.  See :func:`open_econs.uecm_fit`."""
        return uecm_fit(self._data, y, **kwargs)

    def ardl_select_order(self, y: str, **kwargs: Any) -> Any:
        """Select ARDL order by IC.  See :func:`open_econs.ardl_select_order`."""
        return ardl_select_order(self._data, y, **kwargs)

    def __repr__(self) -> str:
        head = f"TimeSeriesContext ({self._data.shape[0]} rows, {self._data.shape[1]} cols)"
        if self._time is not None:
            head += f" [time={self._time!r}"
            if self._freq is not None:
                head += f", freq={self._freq!r}"
            head += "]"
        return head
