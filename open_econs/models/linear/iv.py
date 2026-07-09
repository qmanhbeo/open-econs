from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from open_econs._version import __version__
from open_econs._internal import errors
from open_econs.core.base import BaseModel


class IVResult(BaseModel):
    def __init__(
        self,
        *,
        formula: str,
        nobs: int,
        df_resid: int,
        df_model: int,
        cov_type: str,
        coefficients: pd.Series,
        std_errors: pd.Series,
        z_stats: pd.Series,
        p_values: pd.Series,
        conf_int: pd.DataFrame,
        rsd: float,
        first_stage_f: float,
        first_stage_p: float,
        fitted: pd.Series,
        residuals: pd.Series,
        call: dict[str, Any],
    ) -> None:
        self.formula = formula
        self.data_shape = (nobs, coefficients.shape[0])
        self.cov_type = cov_type
        self.call = call
        self.timestamp = datetime.now()
        self.package_version = __version__

        self.nobs = nobs
        self.df_resid = df_resid
        self.df_model = df_model
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.z_stats = z_stats
        self.p_values = p_values
        self.conf_int = conf_int
        self.rsd = rsd
        self.first_stage_f = first_stage_f
        self.first_stage_p_value = first_stage_p
        self.fitted_values = fitted if fitted is not None else pd.Series(dtype=float)
        self.residuals = residuals

        self._freeze()

    def tidy(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Variable": self.coefficients.index,
            "Coef": self.coefficients.values,
            "Std Err": self.std_errors.values,
            "z": self.z_stats.values,
            "P>|z|": self.p_values.values,
            "0.025": self.conf_int["lower"].values,
            "0.975": self.conf_int["upper"].values,
        })
        df.index.name = None
        return df

    def summary(self) -> str:
        header = (
            f"                       IV-2SLS Regression Results                         \n"
            f"======================================================================\n"
            f"Dep. Variable:               {self.formula.split('~')[0].strip()}\n"
            f"No. Observations:            {self.nobs}\n"
            f"Df Residuals:                {self.df_resid}\n"
            f"Df Model:                    {self.df_model}\n"
            f"First-stage F:               {self.first_stage_f:.4f}\n"
            f"First-stage p-value:         {self.first_stage_p_value:.6e}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return (
            header + tbl +
            "\n======================================================================\n"
        )

    def first_stage(self) -> pd.DataFrame:
        base_columns = [c for c in self.coefficients.index]
        return pd.DataFrame({
            "Variable": base_columns,
            "F": [self.first_stage_f] * len(base_columns),
            "P>F": [self.first_stage_p_value] * len(base_columns),
        })


def iv(
    formula: str,
    data: pd.DataFrame,
    cov_type: str = "nonrobust",
) -> IVResult:
    """Estimate an IV-2SLS regression.

    Parameters
    ----------
    formula : str
        Three-part formula ``y ~ x1 + x2 | z1 + z2`` where variables before
        ``|`` are the endogenous/exogenous regressors and variables after ``|``
        are instruments.
    data : pd.DataFrame
        Data containing all variables referenced in *formula*.
    cov_type : str, default "nonrobust"
        Covariance estimator type.

    Returns
    -------
    IVResult
        Immutable result object with coefficient arrays.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.iv("y ~ x1 + x2 | z1 + z2", data=df)
    >>> r.tidy()
    >>> r.first_stage()
    """
    call = _capture_call(formula=formula, cov_type=cov_type)

    if "|" not in formula:
        raise ValueError(
            "iv() requires a three-part formula: 'y ~ x1 + x2 | z1 + z2'"
        )

    dependent_exog, instruments = formula.split("|", 1)
    y_part = dependent_exog.strip()
    instr_part = instruments.strip()

    from formulaic import Formula
    try:
        dep_matrices = Formula(y_part).get_model_matrix(data, na_action="drop")
    except Exception as e:
        msg = str(e)
        if "not present in the dataset" in msg or "is not present" in msg:
            import re as _re
            m = _re.search(r"`(\w+)`", msg)
            bad_col = m.group(1) if m else y_part
            raise errors.missing_column_error(bad_col, data.columns.tolist()) from e
        raise

    if hasattr(dep_matrices, "rhs"):
        XX = dep_matrices.rhs
        yy = dep_matrices.lhs
    else:
        from open_econs._internal.formula import parse_formula as _parse
        yy, XX = _parse(y_part, data)

    original_n = len(data)
    dropped = original_n - len(yy)
    if dropped > 0:
        import warnings as _w
        _w.warn(
            errors.rows_dropped_warning(dropped, original_n, []),
            RuntimeWarning,
            stacklevel=3,
        )

    if len(yy) == 0:
        raise errors.empty_data_error(original_n, dropped, [])

    y_arr = yy.values.ravel().astype(float)
    X_arr = XX.values.astype(float)

    instr_matrices = Formula(instr_part).get_model_matrix(data, na_action="drop")
    Z = instr_matrices.rhs if hasattr(instr_matrices, "rhs") else instr_matrices
    Z_arr = Z.values.astype(float)

    import statsmodels.api as sm

    first_stage_f = 0.0
    first_stage_p = 0.0

    exog_cols = XX.columns.tolist()

    for idx, col in enumerate(exog_cols):
        x = X_arr[:, idx]
        Z_with_const = sm.add_constant(Z_arr)
        try:
            fs_fit = sm.OLS(x, Z_with_const).fit()
        except Exception:
            continue
        fs_f_stat = fs_fvalue(fs_fit)
        if fs_f_stat > first_stage_f:
            first_stage_f = fs_f_stat
            first_stage_p = float(fs_fit.f_pvalue)

    try:
        from statsmodels.sandbox.regression.gmm import IV2SLS
        fitted = IV2SLS(y_arr, X_arr, Z_arr).fit()
    except Exception as e:
        raise RuntimeError(f"IV2SLS estimation failed: {e}") from e

    coef_arr = fitted.params
    se_arr = fitted.bse
    z_arr = fitted.tvalues
    p_arr = fitted.pvalues
    conf_arr = fitted.conf_int()

    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=XX.columns,
    )

    residuals_arr = fitted.resid
    fitted_arr = y_arr - residuals_arr

    return IVResult(
        formula=formula,
        nobs=int(fitted.nobs),
        df_resid=int(fitted.df_resid),
        df_model=int(fitted.df_model),
        cov_type=cov_type,
        coefficients=pd.Series(coef_arr, index=XX.columns),
        std_errors=pd.Series(se_arr, index=XX.columns),
        z_stats=pd.Series(z_arr, index=XX.columns),
        p_values=pd.Series(p_arr, index=XX.columns),
        conf_int=conf_int,
        rsd=float(np.sqrt(fitted.mse_resid)),
        first_stage_f=first_stage_f,
        first_stage_p=first_stage_p,
        fitted=pd.Series(fitted_arr, index=XX.index, name="fitted"),
        residuals=pd.Series(fitted.resid, index=XX.index, name="residuals"),
        call=call,
    )


def _capture_call(**kwargs: Any) -> dict[str, Any]:
    kwargs["timestamp"] = str(datetime.now())
    kwargs["package_version"] = __version__
    return kwargs


def fs_fvalue(fitted: Any) -> float:
    try:
        return float(fitted.fvalue)
    except (ValueError, AttributeError):
        return float("nan")