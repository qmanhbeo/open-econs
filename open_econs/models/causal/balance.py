import numpy as np
import pandas as pd
from scipy import stats as _stats


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return np.average(x, weights=w)


def _wvar_iw(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted variance matching Stata's ``[iw=w]`` convention.

    ``sum(w * (x - weighted_mean)^2) / (sum(w) - 1)``
    """
    m = _wmean(x, w)
    return np.sum(w * (x - m) ** 2) / (w.sum() - 1)


def _wls_t(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted OLS ``y ~ 1 + x`` with Stata's ``[iw=w]`` convention.

    Returns ``(t_stat, p_value)`` using MSE = WSS / (sum(w) - k)
    and Student's t with sum(w) - k degrees of freedom.
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    k = X.shape[1]

    WX = X * w[:, np.newaxis]
    XWX = X.T @ WX
    XWy = X.T @ (w * y)
    beta = np.linalg.solve(XWX, XWy)

    e = y - X @ beta
    wss = np.sum(w * e ** 2)
    sum_w = w.sum()
    mse = wss / (sum_w - k)
    cov = mse * np.linalg.inv(XWX)
    se = np.sqrt(np.diag(cov))

    t = beta / se
    df = sum_w - k
    p = 2 * _stats.t.sf(np.abs(t), df)
    return float(t[1]), float(p[1])


def balance(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str] | None = None,
    weights: str | None = None,
) -> pd.DataFrame:
    """Covariate balance table for a binary treatment.

    For each covariate, compares means between the treated and control groups.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data.
    treatment : str
        Name of the binary treatment column (must take exactly two values).
    covariates : list of str, optional
        Covariates to compare.  If omitted, all numeric columns other than
        ``treatment`` are used.
    weights : str, optional
        Name of a weight column in ``data``.  When provided, weights are
        applied uniformly to both groups (no treated-weight override).

        The SMD denominator uses **unweighted** full-sample pooled variance
        (Rosenbaum–Rubin, 1985), matching ``pstest``'s ``%bias`` convention.
        The variance ratio uses **weighted** group variances matching
        Stata's ``[iw=w]`` convention (effective N = sum(weights)).
        The t-test comes from a weighted OLS regression of the covariate on
        the treatment indicator, also following Stata's ``[iw=w]`` convention.

        .. note::
           open-econs applies weights uniformly as given — it does *not*
           replicate ``pstest``'s internal override that forces treated
           weights to 1.  This is a deliberate simplification: both of the
           library's current matching estimators (``psm()`` and ``cem()``)
           already produce weight vectors where treated weight = 1 by
           construction, so the two conventions produce identical results
           for every currently anticipated caller.

    The difference is always computed as **Treated Mean − Control Mean**.
    A positive difference means the treated group has a larger sample mean
    for that covariate; a negative difference means the control group does.

    Returns
    -------
    pd.DataFrame
        One row per covariate with treated/control means, the difference,
        standard deviations, the t-statistic and p-value, sorted by p-value
        ascending (largest imbalance first).

        When ``weights`` is provided, ``SMD`` and ``Variance Ratio`` columns
        are also included.
    """
    treatment_vals = data[treatment].unique()
    if len(treatment_vals) != 2:
        raise ValueError(
            f"treatment column '{treatment}' must have exactly 2 unique values, "
            f"got {len(treatment_vals)}"
        )

    treated_val = treatment_vals[1]
    control_val = treatment_vals[0]

    treated = data[data[treatment] == treated_val]
    control = data[data[treatment] == control_val]

    have_weights = weights is not None
    if have_weights:
        weight_col = data[weights].values

    if covariates is None:
        covariates = [
            c
            for c in data.columns
            if c != treatment and c != weights and np.issubdtype(data[c].dtype, np.number)
        ]

    treat_numeric = (data[treatment].values == treated_val).astype(float)

    rows = []
    for var in covariates:
        if have_weights:
            not_nan = ~data[var].isna().values
            is_treat = treat_numeric == 1.0
            t_mask = not_nan & is_treat
            c_mask = not_nan & (~is_treat)

            t_vals = data[var].values[t_mask]
            c_vals = data[var].values[c_mask]
            t_w = weight_col[t_mask]
            c_w = weight_col[c_mask]
            if len(t_vals) < 2 or len(c_vals) < 2:
                continue

            t_mean = _wmean(t_vals, t_w)
            c_mean = _wmean(c_vals, c_w)
            diff = t_mean - c_mean

            t_std = np.sqrt(_wvar_iw(t_vals, t_w))
            c_std = np.sqrt(_wvar_iw(c_vals, c_w))

            t_uvar = np.var(t_vals, ddof=1)
            c_uvar = np.var(c_vals, ddof=1)
            pooled_sd = np.sqrt((t_uvar + c_uvar) / 2)
            smd = diff / pooled_sd

            vr = _wvar_iw(t_vals, t_w) / _wvar_iw(c_vals, c_w)

            stat, pval = _wls_t(
                treat_numeric[not_nan],
                data[var].values[not_nan],
                weight_col[not_nan],
            )

            rows.append({
                "Variable": var,
                "Treated Mean": round(t_mean, 4),
                "Control Mean": round(c_mean, 4),
                "Difference": round(diff, 4),
                "Treated Std": round(t_std, 4),
                "Control Std": round(c_std, 4),
                "SMD": round(smd, 4),
                "Variance Ratio": round(vr, 4),
                "t-statistic": round(stat, 4),
                "P>|t|": round(pval, 4),
            })
        else:
            t_vals = treated[var].dropna().values
            c_vals = control[var].dropna().values
            if len(t_vals) < 2 or len(c_vals) < 2:
                continue

            t_mean = t_vals.mean()
            c_mean = c_vals.mean()
            t_std = t_vals.std(ddof=1)
            c_std = c_vals.std(ddof=1)
            diff = t_mean - c_mean

            stat, pval = _stats.ttest_ind(t_vals, c_vals, equal_var=False)

            rows.append({
                "Variable": var,
                "Treated Mean": round(t_mean, 4),
                "Control Mean": round(c_mean, 4),
                "Difference": round(diff, 4),
                "Treated Std": round(t_std, 4),
                "Control Std": round(c_std, 4),
                "t-statistic": round(stat, 4),
                "P>|t|": round(pval, 4),
            })

    return pd.DataFrame(rows).sort_values("P>|t|").reset_index(drop=True)
