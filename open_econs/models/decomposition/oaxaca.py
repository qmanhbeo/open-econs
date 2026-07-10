
import numpy as np
import pandas as pd
from statsmodels.stats.oaxaca import OaxacaBlinder

from open_econs.core.call_capture import capture_call as _capture_call
from open_econs._internal import errors
from open_econs._internal.formula import parse_formula
from open_econs.core.results import OaxacaResult


_REFERENCE_ALIASES = {
    "pooled": "pooled",
    "omega": "nuemark",
    "group1": "self_submitted",
    "group2": "self_submitted",
}


def _resolve_reference(reference: str | float) -> tuple[str, float]:
    """Convert a user-facing reference spec to a statsmodels two_fold_type + weight."""
    if isinstance(reference, (int, float)):
        return "self_submitted", float(reference)
    ref = reference.lower()
    if ref not in _REFERENCE_ALIASES:
        allowed = ", ".join(f"'{k}'" for k in _REFERENCE_ALIASES)
        raise ValueError(
            f"Unknown reference '{reference}'. Allowed: {allowed}, or a float (0-1)."
        )
    sm_type = _REFERENCE_ALIASES[ref]
    weight = 1.0 if ref == "group1" else 0.0
    return sm_type, weight


def oaxaca(
    formula: str,
    data: pd.DataFrame,
    by: str,
    decomposition_type: str = "two-fold",
    reference: str | float = "pooled",
    reverse: bool = False,
    std: bool = False,
    bootstrap_n: int = 1000,
    conf_level: float = 0.95,
    seed: int | None = None,
    swap: bool = True,
) -> OaxacaResult:
    """Perform an Oaxaca-Blinder decomposition.

    Parameters
    ----------
    formula : str
        Two-sided formula string, e.g. ``"income ~ education + age + female"``.
        ``by`` must appear on the RHS of the formula.
    data : pd.DataFrame
        Data containing all variables referenced in *formula*.
    by : str
        Column name of the binary group indicator.  Must appear in the RHS
        of *formula*.
    decomposition_type : str, default "two-fold"
        Either ``"two-fold"`` or ``"three-fold"``.
    reference : str or float, default "pooled"
        Controls the reference coefficients used in the ``"two-fold"``
        decomposition.  Options:

        - ``"pooled"`` — pooled model including the group dummy (Stata's ``pooled``).
        - ``"omega"`` — pooled model *excluding* the group dummy (Stata's ``omega``).
        - ``"group1"`` — Group 1 coefficients as reference (Stata's ``weight(1)``).
        - ``"group2"`` — Group 2 coefficients as reference (Stata's ``weight(0)``).
        - float (0–1) — custom weight for Group 1 coefficients
          (1 = Group 1, 0 = Group 2; Stata's ``weight()``).

        Ignored for ``"three-fold"``.
    reverse : bool, default False
        If ``True`` and ``decomposition_type="three-fold"``, uses Group 1
        coefficients as the reference (Stata's ``threefold(reverse)``).
        The default (``False``) uses Group 2 coefficients as the reference.
    std : bool, default False
        If True, compute bootstrapped standard errors.  Bootstrap is
        computationally expensive; 500 iterations is a reasonable minimum
        for exploration, 1000+ for publication.
    bootstrap_n : int, default 1000
        Number of bootstrap replications (only used when ``std=True``).
        Statsmodels' own default is 5000; 1000 is a reasonable trade-off
        for exploration, 5000+ for publication-grade estimates.
    conf_level : float, default 0.95
        Confidence level for trimming extreme bootstrap draws.
    seed : int, optional
        Random seed for reproducible bootstrap standard errors.
    swap : bool, default True
        If ``True`` (default), ensures ``total_gap >= 0`` by swapping the
        internal group ordering when the gap would otherwise be negative.
        If ``False``, uses the natural statsmodels ordering (group 1 minus
        group 0, where group 0 is the smaller value of the binary ``by``
        column). When the gap is already positive, ``swap=True`` and
        ``swap=False`` produce identical results. ``swap`` only affects sign
        — it does not reverse the decomposition direction.

    Terminology (Stata ``oaxaca`` ↔ open-econs ``OaxacaResult``)
    ---------------------------------------------------------
    Stata's ``e(b)`` uses different names for the same decomposition
    components depending on the decomposition type:

    ================== ==================== ==========================
    Stata ``e(b)``     OE attribute         Applies to
    ================== ==================== ==========================
    ``gap``            ``.total_gap``       all variants
    ``explained``      ``.explained``       two-fold
    ``unexplained``    ``.unexplained``     two-fold
    ``endowment``      ``.explained``       three-fold
    ``coefficients``   ``.unexplained``     three-fold
    ``interaction``    ``.interaction``     three-fold
    ================== ==================== ==========================

    For three-fold decompositions, ``.explained`` corresponds to the
    endowment effect and ``.unexplained`` to the coefficient effect,
    *not* the two-fold meanings of those terms.

    Notes
    -----
    Stata reference implementation: ``oaxaca`` package by Ben Jann — see
    the `repository <https://github.com/benjann/oaxaca>`_.

    Returns
    -------
    OaxacaResult
        Immutable result object with ``.explained``, ``.unexplained``,
        ``.total_gap``, and ``.tidy()``.

    Examples
    --------
    >>> import open_econs as oe
    >>> result = oe.oaxaca("income ~ education + age + female", data=df, by="female")
    >>> result.tidy()
    """
    if by not in data.columns:
        raise errors.missing_column_error(by, data.columns.tolist())

    unique_vals = data[by].dropna().unique()
    if len(unique_vals) != 2:
        raise errors.non_binary_error(by, unique_vals)

    call = _capture_call(
        formula=formula, by=by, decomposition_type=decomposition_type,
        reference=reference, reverse=reverse,
        std=std, bootstrap_n=bootstrap_n, conf_level=conf_level, seed=seed,
        swap=swap,
    )

    yy, XX = parse_formula(formula, data)
    y_arr = yy.values.ravel()

    if by in XX.columns:
        bifurcate_idx = list(XX.columns).index(by)
    else:
        # formulaic may encode C(by) as one-hot columns; detect and collapse
        # to a single binary column for the Oaxaca bifurcation.
        patterns = [f"C({by})[", f"{by}["]
        encoded_cols = [
            c for c in XX.columns
            if any(c.startswith(p) for p in patterns)
        ]
        if encoded_cols:
            from pandas import get_dummies
            dummies = get_dummies(data.loc[XX.index, by], prefix=by, drop_first=True)
            if len(dummies.columns) == 1:
                XX = XX.copy()
                for c in encoded_cols:
                    XX.drop(columns=[c], inplace=True)
                cname = str(by)
                XX.insert(XX.shape[1], cname, dummies.iloc[:, 0].values.astype(float))
                bifurcate_idx = list(XX.columns).index(cname)
            else:
                raise ValueError(
                    f"The 'by' column '{by}' has {len(dummies.columns) + 1} unique "
                    f"values (must be exactly 2 for Oaxaca decomposition)."
                )
        else:
            raise ValueError(
                f"Column '{by}' not found in the design matrix. Ensure '{by}' "
                f"appears on the RHS of the formula (e.g. '{formula}'). "
                f"Available columns in design matrix: {list(XX.columns)}"
            )
    hasconst = any("Intercept" in c for c in XX.columns)

    if seed is not None:
        np.random.seed(seed)

    model = OaxacaBlinder(
        y_arr,
        XX.values,
        bifurcate=bifurcate_idx,
        hasconst=hasconst,
        swap=swap,
    )

    if decomposition_type == "two-fold":
        two_fold_type, weight = _resolve_reference(reference)
        submitted_weight = weight if two_fold_type == "self_submitted" else None
        stats_result = model.two_fold(two_fold_type=two_fold_type, submitted_weight=submitted_weight, std=std, n=bootstrap_n, conf=conf_level)
        explained = float(stats_result.params[1])
        unexplained = float(stats_result.params[0])
        gap_val = float(stats_result.params[2])
        interaction = None
    elif decomposition_type == "three-fold":
        if reverse:
            gap_val = float(model.gap)
            f_mean = model.exog_f_mean
            s_mean = model.exog_s_mean
            f_params = model._f_model.params
            s_params = model._s_model.params
            endowment_full = float((f_mean - s_mean) @ f_params)
            coeff_full = float(f_mean @ (f_params - s_params))
            interaction_full = -float((f_mean - s_mean) @ (f_params - s_params))
            explained = endowment_full
            unexplained = coeff_full
            interaction = interaction_full
            stats_result = None
        else:
            stats_result = model.three_fold(std=std, n=bootstrap_n, conf=conf_level)
            endowment = float(stats_result.params[0])
            coefficients = float(stats_result.params[1])
            interaction = float(stats_result.params[2])
            gap_val = float(stats_result.params[3])
            explained = endowment
            unexplained = coefficients
    else:
        raise ValueError(
            f"Unknown decomposition_type '{decomposition_type}'. "
            f"Use 'two-fold' or 'three-fold'."
        )

    var_names = [c for c in XX.columns if c != by]
    f_params = model._f_model.params
    s_params = model._s_model.params
    f_mean = model.exog_f_mean
    s_mean = model.exog_s_mean

    if decomposition_type == "two-fold":
        ref_params = model.t_params
        endow_vec = (f_mean - s_mean) * ref_params
        unexpl_vec = f_mean * (f_params - ref_params) + s_mean * (ref_params - s_params)
        var_detail = pd.DataFrame({
            "Variable": var_names,
            "Explained": endow_vec[:len(var_names)],
            "Unexplained": unexpl_vec[:len(var_names)],
        })
    else:
        if reverse:
            endow_vec = (f_mean - s_mean) * f_params
            coeff_vec = f_mean * (f_params - s_params)
            inter_vec = -(f_mean - s_mean) * (f_params - s_params)
        else:
            endow_vec = (f_mean - s_mean) * s_params
            coeff_vec = s_mean * (f_params - s_params)
            inter_vec = (f_mean - s_mean) * (f_params - s_params)
        var_detail = pd.DataFrame({
            "Variable": var_names,
            "Endowment": endow_vec[:len(var_names)],
            "Coefficients": coeff_vec[:len(var_names)],
            "Interaction": inter_vec[:len(var_names)],
        })

    unique_vals_sorted = sorted(data[by].dropna().unique(), key=str)
    idx_map = {float(i): str(v) for i, v in enumerate(unique_vals_sorted)}
    group_labels = (idx_map.get(model.bi[0], str(model.bi[0])),
                    idx_map.get(model.bi[1], str(model.bi[1])))

    std_series: pd.Series | None = None
    if std and stats_result is not None and stats_result.std is not None:
        if decomposition_type == "two-fold":
            labels = ["Unexplained", "Explained"]
        else:
            labels = ["Endowment", "Coefficients", "Interaction"]
        std_series = pd.Series(stats_result.std, index=labels, name="std_err")

    return OaxacaResult(
        formula=formula,
        nobs=int(len(yy)),
        n_params=XX.shape[1],
        cov_type="nonrobust",
        explained=explained,
        unexplained=unexplained,
        interaction=interaction,
        total_gap=gap_val,
        decomposition_type=decomposition_type,
        by_groups=group_labels,
        std=std_series,
        call=call,
        variable_detail=var_detail,
    )


