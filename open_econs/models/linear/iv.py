from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from open_econs._version import __version__
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs.core.base import BaseModel
from open_econs.core.cov_type import validate_cov_type


_IV_COV_MAP = {
    "nonrobust": "unadjusted",
    "HC0": "robust",
    "HC1": "robust",
    "HC2": "robust",
    "HC3": "robust",
    "robust": "robust",
    "heteroskedastic": "robust",
    "unadjusted": "unadjusted",
    "homoskedastic": "unadjusted",
    "kernel": "kernel",
    "clustered": "clustered",
}

_IV_DEBIAS_MAP = {
    "nonrobust": False,
    "HC0": False,
    "HC1": True,
    "HC2": False,
    "HC3": False,
    "robust": False,
    "unadjusted": False,
    "homoskedastic": False,
}


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
        first_stage_f: pd.Series,
        cragg_donald_f: float,
        hansen_j_stat: float,
        hansen_j_p: float,
        fitted: pd.Series,
        residuals: pd.Series,
        call: dict[str, Any],
        _fit: Any = None,
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
        self.cragg_donald_stat = cragg_donald_f
        self.hansen_j_stat = hansen_j_stat
        self.hansen_j_p_value = hansen_j_p
        self.fitted_values = fitted if fitted is not None else pd.Series(dtype=float)
        self.residuals = residuals
        self._fit = _fit

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
        endog_name = self.formula.split("~")[0].strip()
        parts = self.formula.split("|", 1)
        instr_raw = parts[1].strip()
        fs_rows = []
        for var, fval in self.first_stage_f.items():
            fs_rows.append(f"  {var:<20s}  {fval:>8.4f}")

        fs_block = "\n".join(fs_rows)
        header = (
            f"                       IV-2SLS Regression Results                         \n"
            f"======================================================================\n"
            f"Dep. Variable:               {endog_name}\n"
            f"Exogenous:                   {self._exog_names()}\n"
            f"Endogenous:                  {self._endog_names()}\n"
            f"Instruments:                 {instr_raw}\n"
            f"No. Observations:            {self.nobs}\n"
            f"Df Residuals:                {self.df_resid}\n"
            f"Df Model:                    {self.df_model}\n"
            f"Covariance Type:             {self.cov_type}\n"
            f"======================================================================\n"
            f"First-stage F-statistics:\n{fs_block}\n"
            f"======================================================================\n"
            f"Cragg-Donald Wald F-stat:   {self.cragg_donald_stat:.4f}\n"
            f"Hansen J (overid) chi2:      {self.hansen_j_stat:.4f}\n"
            f"Hansen J p-value:            {self.hansen_j_p_value:.6e}\n"
            f"======================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return (
            header + tbl +
            "\n======================================================================\n"
        )

    def _exog_names(self) -> str:
        if self._fit is None:
            return "N/A"
        return ", ".join(self._fit.model.exog.cols) if self._fit.model.exog else "none"

    def _endog_names(self) -> str:
        if self._fit is None:
            return "N/A"
        return ", ".join(self._fit.model.endog.cols)

    def vcov(self) -> pd.DataFrame:
        """Return the 2SLS/IV parameter variance-covariance matrix as a DataFrame."""
        if self._fit is None:
            raise RuntimeError(
                "vcov() requires a fitted model result."
            )
        return pd.DataFrame(
            self._fit.cov,
            index=self.coefficients.index,
            columns=self.coefficients.index,
        )

    def first_stage(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Variable": self.first_stage_f.index,
            "F": self.first_stage_f.values,
        })


def iv(
    formula: str,
    data: pd.DataFrame,
    cov_type: str = "robust",
    lags: int | None = None,
    time: str | None = None,
    hac_adjust: bool = False,
) -> IVResult:
    """Estimate an IV-2SLS regression.

    Parameters
    ----------
    formula : str
        Three-part formula ``y ~ x1 + w1 + w2 | x1 ~ z1 + z2``
        where variables left of ``~`` inside the RHS block are endogenous,
        and variables left of ``|`` but outside the ``~`` block are exogenous.

        Alternatively, the simpler ``y ~ x1 + w1 + w2 | z1 + z2`` form
        is accepted where **all** non-LHS variables are treated as endogenous
        (legacy, not recommended for applied use).

        Best practice: ``y ~ w1 + w2 | x1 ~ z1 + z2`` where ``w1, w2`` are
        exogenous controls, ``x1`` is the endogenous regressor, and
        ``z1, z2`` are instruments.
    data : pd.DataFrame
        Data containing all variables referenced in *formula*.
    cov_type : str, default "robust"
        Covariance estimator type. Mapped to linearmodels convention:
        ``"nonrobust"`` -> ``"unadjusted"``, ``"HC1"`` -> ``"robust"``
        with debiased=True, ``"HC0"/"HC2"/"HC3"`` -> ``"robust"``.

        Set ``cov_type="HAC"`` to use Newey-West (1987) heteroskedasticity-
        and autocorrelation-robust standard errors with a Bartlett kernel;
        the number of lags is given by *lags* and the time ordering by *time*.
    lags : int, optional
        Number of lags for Newey-West HAC (required when ``cov_type="HAC"``).
    time : str, optional
        Column with the time index used to order observations for Newey-West
        HAC.  Observations are sorted by this column before fitting.
    hac_adjust : bool, default False
        Degrees-of-freedom correction for Newey-West HAC standard errors.
        When ``True``, the HAC variance is multiplied by ``N / (N - K)``,
        matching Stata's ``ivregress`` default behavior.

    Returns
    -------
    IVResult
        Immutable result object with coefficient arrays, weak-instrument
        diagnostics (Cragg-Donald Wald F-stat), and overidentification test
        (Hansen J statistic).

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.iv("y ~ w1 + w2 | x1 ~ z1 + z2", data=df)
    >>> r.tidy()
    >>> r.first_stage()
    >>> r.cragg_donald_stat
    """
    call = _capture_call(
        formula=formula, cov_type=cov_type, lags=lags, time=time,
        hac_adjust=hac_adjust,
    )

    cov_type = validate_cov_type(
        cov_type,
        accepted=set(_IV_COV_MAP.keys()) | {"HAC"},
        estimator="iv()",
    )

    if cov_type == "HAC":
        if lags is None:
            raise ValueError(
                "Newey-West HAC requires `lags` (e.g. lags=1)."
            )
        if time is not None:
            data = data.sort_values(time)

    parsed = _parse_iv_formula(formula, data)
    y_arr = parsed["y"]
    X_full = parsed["X"]
    has_inner_endog = parsed["has_inner_endog"]
    instr_matrix = parsed["instr_matrix"]
    exog_idx = parsed["exog_idx"]
    endog_idx = parsed["endog_idx"]

    if not has_inner_endog:
        import warnings as _w
        _w.warn(
            "The legacy IV syntax 'y ~ rhs | instruments' treats ALL RHS variables "
            "as endogenous. This is almost certainly wrong if you have exogenous "
            "controls. Use the new syntax: 'y ~ exog | endog ~ instruments'. "
            "See the iv() docstring for details.",
            FutureWarning, stacklevel=3,
        )

    Z_arr = instr_matrix if instr_matrix.shape[1] > 0 else None

    X_exog = X_full[:, exog_idx] if exog_idx else None
    X_endog = X_full[:, endog_idx] if endog_idx else None

    from linearmodels.iv import IV2SLS as LM_IV2SLS
    try:
        if cov_type == "HAC":
            fitted = LM_IV2SLS(y_arr, X_exog, X_endog, Z_arr).fit(
                cov_type="kernel",
                debiased=hac_adjust,
                bandwidth=lags,
            )
            cov_label = f"HAC({lags})"
        else:
            fitted = LM_IV2SLS(y_arr, X_exog, X_endog, Z_arr).fit(
                cov_type=_IV_COV_MAP.get(cov_type, "robust"),
                debiased=_IV_DEBIAS_MAP.get(cov_type, False),
            )
            cov_label = cov_type
    except Exception as e:
        raise RuntimeError(f"IV2SLS estimation failed: {e}") from e

    coef_arr = fitted.params.values
    se_arr = fitted.std_errors.values
    z_arr = fitted.tstats.values
    p_arr = fitted.pvalues.values
    conf_arr = fitted.conf_int(level=0.95).values

    coef_names = fitted.params.index.tolist()
    conf_int = pd.DataFrame(
        {"lower": conf_arr[:, 0], "upper": conf_arr[:, 1]},
        index=coef_names,
    )

    residuals_arr = fitted.resids.values
    fitted_arr = y_arr - residuals_arr

    fs_f_stats = {}
    for en_name in fitted.model.endog.cols:
        fs = cast(Any, fitted).first_stage
        if fs is not None and en_name in fs.individual:
            ind_res = fs.individual[en_name]
            f_stat = ind_res.f_statistic.stat if hasattr(ind_res, "f_statistic") else float("nan")
            fs_f_stats[en_name] = float(f_stat)
        else:
            fs_f_stats[en_name] = float("nan")

    fs_f_series = pd.Series(fs_f_stats, name="F")

    try:
        cragg_donald = float(min(fs_f_stats.values())) if fs_f_stats else float("nan")
    except Exception:
        cragg_donald = float("nan")

    try:
        overid = cast(Any, fitted).sargan
        hansen_j = float(overid.stat)
        hansen_p = float(overid.pval)
    except (AttributeError, Exception):
        hansen_j = float("nan")
        hansen_p = float("nan")

    return IVResult(
        formula=formula,
        nobs=int(fitted.nobs),
        df_resid=int(fitted.df_resid),
        df_model=int(fitted.df_model),
        cov_type=cov_label,
        coefficients=pd.Series(coef_arr, index=coef_names),
        std_errors=pd.Series(se_arr, index=coef_names),
        z_stats=pd.Series(z_arr, index=coef_names),
        p_values=pd.Series(p_arr, index=coef_names),
        conf_int=conf_int,
        rsd=float(np.sqrt(fitted.s2)) if hasattr(fitted, "s2") else float("nan"),
        first_stage_f=fs_f_series,
        cragg_donald_f=cragg_donald,
        hansen_j_stat=hansen_j,
        hansen_j_p=hansen_p,
        fitted=pd.Series(fitted_arr, index=parsed["index"], name="fitted"),
        residuals=pd.Series(residuals_arr, index=parsed["index"], name="residuals"),
        call=call,
        _fit=fitted,
    )


def _extract_vars(expr: str) -> list[str]:
    import re
    terms = re.split(r"[+\-*/]", expr)
    result = []
    for t in terms:
        t = t.strip()
        if t and t not in ("1", "0", "-1"):
            result.append(t)
    return result


def _parse_iv_formula(formula: str, data: pd.DataFrame) -> dict:
    """Parse an IV/2SLS/GMM formula into aligned model matrices.

    Shared by :func:`iv` and :func:`open_econs.models.linear.gmm.gmm` so the
    ``y ~ exog | endog ~ instruments`` grammar is implemented exactly once.

    Returns a dict with:

    * ``y`` : ndarray (n,) dependent vector.
    * ``X`` : ndarray (n, p) full regressor matrix (incl. intercept).
    * ``coef_names`` : list[str] of length ``p`` (X column labels).
    * ``exog_idx`` : positions of exogenous regressors (incl. intercept) in ``X``.
    * ``endog_idx`` : positions of endogenous regressors in ``X``.
    * ``instr_matrix`` : ndarray (n, L_instr) instrument variables only (intercept
      dropped) -- the explicit instruments named after ``~`` in the inner block.
    * ``has_inner_endog`` : whether the ``| endog ~ instruments`` syntax was used.
    * ``endog_vars`` : list of endogenous variable names.
    * ``index`` : pandas.Index of the kept (aligned, non-missing) rows.
    * ``dropped`` / ``original_n`` : row accounting.

    The instrument matrix ``Z`` handed to a GMM solver must be the exogenous
    regressors (their own instruments) concatenated with ``instr_matrix``;
    the caller assembles that.  Rows of ``y``, ``X`` and ``instr_matrix`` are
    aligned to a common index so they never silently disagree.
    """
    if "|" not in formula:
        raise ValueError(
            "This estimator requires a three-part formula: 'y ~ exog | endog ~ instruments'"
        )

    y_part, instr_part = formula.split("|", 1)
    y_part = y_part.strip()
    instr_part = instr_part.strip()

    rhs_split = y_part.split("~", 1)
    dep_var = rhs_split[0].strip()
    rhs_expr = rhs_split[1].strip() if len(rhs_split) > 1 else ""

    has_inner_endog = "~" in instr_part
    if has_inner_endog:
        endog_part, instr_expr = instr_part.split("~", 1)
        endog_expr = endog_part.strip()
        instr_expr = instr_expr.strip()
        endog_vars = _extract_vars(endog_expr)
        exog_vars = _extract_vars(rhs_expr)
        all_rhs_vars = list(dict.fromkeys(exog_vars + endog_vars))
        full_rhs_formula = " + ".join(all_rhs_vars)
        y_formula = f"{dep_var} ~ {full_rhs_formula}"
    else:
        all_rhs_vars = _extract_vars(rhs_expr)
        endog_vars = all_rhs_vars
        exog_vars = []
        instr_expr = instr_part
        full_rhs_formula = " + ".join(all_rhs_vars)
        y_formula = f"{dep_var} ~ {full_rhs_formula}"

    from formulaic import Formula

    try:
        matrices = Formula(y_formula).get_model_matrix(data, na_action="drop")
    except Exception as e:
        msg = str(e)
        if "not present in the dataset" in msg or "is not present" in msg:
            import re as _re
            m = _re.search(r"`(\w+)`", msg)
            bad_col = m.group(1) if m else y_part
            raise errors.missing_column_error(bad_col, data.columns.tolist()) from e
        raise

    if hasattr(matrices, "rhs"):
        XX = matrices.rhs
        yy = matrices.lhs
    else:
        from open_econs._internal.formula import parse_formula as _parse
        yy, XX = _parse(y_formula, data)
        matrices = None

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

    x_index = XX.index
    instr_matrices = Formula(instr_expr).get_model_matrix(
        data.loc[x_index], na_action="drop",
    )
    # A single-sided instrument formula ("z1 + z2") returns a bare DataFrame;
    # a two-sided one returns a ModelMatrix with .rhs/.lhs.
    Z_instr = instr_matrices.rhs if hasattr(instr_matrices, "rhs") else instr_matrices

    # Align instrument rows to the regressor rows (instruments may carry their
    # own missing values), so y / X / instruments are always the same length.
    keep = x_index.intersection(Z_instr.index)
    yy = yy.loc[keep]
    XX = XX.loc[keep]
    Z_instr = Z_instr.loc[keep]

    y_arr = yy.values.ravel().astype(float)
    X_full = XX.values.astype(float)
    all_cols = XX.columns.tolist()
    instr_cols = [c for c in Z_instr.columns if c != "Intercept"]
    instr_matrix = (
        Z_instr[instr_cols].values.astype(float) if instr_cols
        else np.zeros((len(X_full), 0))
    )

    endog_cols_in_model = [c for c in all_cols if c in endog_vars]
    exog_idx = [i for i, c in enumerate(all_cols) if c not in endog_cols_in_model]
    endog_idx = [i for i, c in enumerate(all_cols) if c in endog_cols_in_model]

    return {
        "y": y_arr,
        "X": X_full,
        "coef_names": all_cols,
        "exog_idx": exog_idx,
        "endog_idx": endog_idx,
        "instr_matrix": instr_matrix,
        "has_inner_endog": has_inner_endog,
        "endog_vars": endog_vars,
        "index": keep,
        "dropped": original_n - len(keep),
        "original_n": original_n,
    }


