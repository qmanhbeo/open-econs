import numpy as np
import pandas as pd
from scipy import stats as _stats


def balance(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    """Covariate balance table for a binary treatment.

    For each covariate, compares means between the treated and control groups
    with a Welch (unequal-variance) two-sample t-test.  This is the standard
    pre-treatment balance check that should accompany any DiD / matching /
    RCT analysis: large, significant differences indicate the groups are not
    comparable and the design is at risk.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data.
    treatment : str
        Name of the binary treatment column (must take exactly two values).
    covariates : list of str, optional
        Covariates to compare.  If omitted, all numeric columns other than
        ``treatment`` are used.

    Returns
    -------
    pd.DataFrame
        One row per covariate with treated/control means, the difference,
        standard deviations, the t-statistic and p-value, sorted by p-value
        ascending (largest imbalance first).
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

    if covariates is None:
        covariates = [c for c in data.columns if c != treatment and np.issubdtype(data[c].dtype, np.number)]

    rows = []
    for var in covariates:
        t_vals = treated[var].dropna()
        c_vals = control[var].dropna()
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