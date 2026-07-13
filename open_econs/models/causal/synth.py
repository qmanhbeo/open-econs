"""Synthetic control (Abadie-Diamond-Hainmueller) core point estimator (public API).

This implements the **core point estimator only**: it fits the donor weights
``W`` and predictor weights ``V`` and reports the synthetic-counterfactual gap
path.  Placebo-in-space / placebo-in-time inference, plotting, and ``predict()``
are intentionally out of scope for this pass (they raise ``NotImplementedError``
and are a separate, later scoped task).

The nested optimization faithfully mirrors the R ``Synth`` package (v1.1-10),
whose source was read during implementation:

* Predictors are standardized per predictor by ``1 / sd`` with ``ddof=1``
  (R's ``sqrt(var(...))`` convention), where the sd is computed across **all**
  units (treated + donors).
* Inner (donor-weight) problem, given ``V``: minimize
  ``(X1 - X0' W)' V (X1 - X0' W)`` subject to ``W >= 0`` and ``sum(W) = 1``.
  R solves this with ``kernlab::ipop`` (a quadratic program); we use
  :func:`scipy.optimize.minimize` (SLSQP), which recovers the same unique QP
  minimizer.
* Outer (predictor-weight) problem: minimize the pre-treatment outcome MSPE
  ``loss.v = mean((Z1 - Z0' W)^2)`` over the predictor-weight simplex, run from
  **two starts** exactly like R ``Synth`` -- (1) equal weights ``1/k`` and
  (2) a regression-derived start ``Beta = (Xall' Xall)^-1 Xall' Zall``,
  ``V = Beta[-1,] %*% t(Beta[-1,])``, ``SV2 = diag(V) / sum(diag(V))`` -- keeping
  whichever converges to the lower ``loss.v``.  (When there is only a single
  predictor, or when ``predictor_weights=`` is supplied, the outer loop is
  skipped and ``V`` is fixed, again matching R's ``custom.v`` path.)
* The objective landscape can be nonconvex in ``V``; SLSQP may therefore land on
  a *different* local optimum of ``V`` than R's ``optimx`` sequence even when the
  two-start procedure is replicated.  This is expected and acceptable: what
  matters for a valid parity claim is that ``W``, the pre-treatment MSPE, and the
  gap path closely agree (which they do, because the inner W problem is a unique
  convex QP).  Do not chase exact ``V`` parity if it does not materialize.

This is new, additive code only: no existing estimator is modified.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from open_econs._internal import errors
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.core.results import SynthResult


def synth(
    data: pd.DataFrame,
    outcome: str,
    treated_unit: Any,
    donor_pool: Sequence[Any],
    *,
    entity: str = "entity",
    time: str = "time",
    pre_period: Any,
    post_period: Any,
    predictors: Optional[Sequence[str]] = None,
    predictor_weights: Optional[Union[pd.Series, Sequence[float]]] = None,
    **solver_kwargs: Any,
) -> SynthResult:
    """Fit a synthetic control (Abadie-Diamond-Hainmueller) point estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.  Must contain the outcome, an entity id column, and a
        time id column.  The panel must be **balanced** over the treated unit and
        the donor pool across the periods used (no missing outcome values for
        those unit/period cells).
    outcome : str
        Name of the outcome column to be fitted / counterfactually predicted.
    treated_unit : Any
        Identifier (value in *entity*) of the single treated unit.
    donor_pool : sequence of Any
        Identifiers (values in *entity*) of the control/donor units.  Must be
        non-empty, contain at least two units, and must **not** include the
        treated unit.
    entity : str, default "entity"
        Name of the column identifying units.
    time : str, default "time"
        Name of the column identifying time periods.
    pre_period : Any
        The **last** pre-treatment period (inclusive).  All periods ``<=`` this
        value are used to fit the synthetic control (predictors + pre-MSPE).
    post_period : Any
        The **first** post-treatment period (inclusive).  All periods ``>=`` this
        value form the post-treatment window in the gap path / post-MSPE.
    predictors : sequence of str, optional
        Explicit covariate columns used as predictors.  Each is aggregated to a
        single predictor by its **mean over the pre-treatment periods** (matching
        R ``Synth``'s default ``predictors.op = "mean"``).  If ``None`` (default),
        the predictors are the outcome's own pre-treatment path -- one predictor
        per pre-treatment period -- the canonical ADH default.
    predictor_weights : pd.Series or sequence of float, optional
        Fixed predictor weights ``V`` (mirrors R ``Synth``'s ``custom.v``).  When
        supplied, the outer ``V`` optimization is skipped entirely and only the
        inner donor-weight ``W`` problem is solved.
    **solver_kwargs
        Extra keyword arguments forwarded verbatim to
        :func:`scipy.optimize.minimize` (e.g. ``maxiter``, ``ftol``).  Applied to
        both the inner (``W``) and outer (``V``) SLSQP solves.

    Returns
    -------
    SynthResult
        Immutable result exposing ``weights`` (donor weights ``W``),
        ``predictor_weights`` (``V``), ``pre_mspe`` / ``post_mspe``, ``gap_path``
        (treated / synthetic / gap over the analysis window), and convergence
        diagnostics straight from the nested ``scipy.optimize.minimize`` calls.

    Notes
    -----
    **Parity scope.**  The estimator is built to match R ``Synth`` (primary
    reference) and is checked against Stata ``synth`` (secondary).  Because the
    inner ``W`` problem is a unique convex QP, ``W`` / pre-MSPE / gap path agree
    closely with R; ``V`` itself may differ slightly due to nonconvex optimizer
    paths and is reported honestly rather than forced to match.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # ── column existence validation ─────────────────────────────────
    for col in (outcome, entity, time):
        if col not in data.columns:
            raise errors.missing_column_error(col, data.columns.tolist())

    # ── donor pool / treated unit validation ────────────────────────
    donor_pool = list(donor_pool)
    all_entities = set(pd.unique(data[entity]))
    if treated_unit not in all_entities:
        raise ValueError(
            f"treated_unit={treated_unit!r} is not present in the '{entity}' "
            f"column. Observed entities include: {sorted(all_entities)[:10]}..."
        )
    if len(donor_pool) == 0:
        raise ValueError("donor_pool must contain at least one control unit.")
    if len(donor_pool) < 2:
        raise ValueError(
            "donor_pool must contain at least two control units (a synthetic "
            "control needs >= 2 donors to form a convex combination)."
        )
    for d in donor_pool:
        if d not in all_entities:
            raise ValueError(
                f"donor {d!r} in donor_pool is not present in the '{entity}' "
                f"column."
            )
    if treated_unit in donor_pool:
        raise ValueError(
            f"treated_unit={treated_unit!r} must not appear in donor_pool "
            f"(a unit cannot be its own donor)."
        )

    if predictors is not None:
        predictors = list(predictors)
        for p in predictors:
            if p not in data.columns:
                raise errors.missing_column_error(p, data.columns.tolist())
        if len(predictors) == 0:
            raise ValueError("predictors must be a non-empty sequence or None.")

    # ── period validation ───────────────────────────────────────────
    all_times = list(pd.unique(data[time]))
    try:
        sorted_times = sorted(all_times)
    except TypeError as e:
        raise ValueError(
            f"Time values in '{time}' are not orderable; synthetic control "
            f"requires comparable time periods."
        ) from e
    if pre_period not in all_times:
        raise ValueError(
            f"pre_period={pre_period!r} is not present in the '{time}' column. "
            f"Observed periods include: {sorted_times[:10]}..."
        )
    if post_period not in all_times:
        raise ValueError(
            f"post_period={post_period!r} is not present in the '{time}' column. "
            f"Observed periods include: {sorted_times[:10]}..."
        )
    i_pre = sorted_times.index(pre_period)
    i_post = sorted_times.index(post_period)
    if i_pre >= i_post:
        raise ValueError(
            f"pre_period={pre_period!r} must come before post_period="
            f"{post_period!r} (got indices {i_pre} >= {i_post} in the sorted "
            f"time axis)."
        )

    pre_times = [t for t in sorted_times if t <= pre_period]
    post_times = [t for t in sorted_times if t >= post_period]
    if len(pre_times) == 0:
        raise ValueError("No pre-treatment periods selected; check pre_period.")
    if len(post_times) == 0:
        raise ValueError("No post-treatment periods selected; check post_period.")

    # ── build the balanced wide outcome matrix ──────────────────────
    try:
        wide = data.pivot(index=entity, columns=time, values=outcome)
    except ValueError as e:
        raise ValueError(
            "The panel is not balanced: the (entity, time) pairs are not unique. "
            "Synthetic control requires exactly one outcome value per "
            f"(entity, time). Underlying error: {e}"
        ) from e
    wide = wide.astype(float)

    if treated_unit not in wide.index:
        raise ValueError(f"treated_unit={treated_unit!r} missing from the pivoted panel.")
    missing_donors = [d for d in donor_pool if d not in wide.index]
    if missing_donors:
        raise ValueError(f"Donor units missing from panel: {missing_donors}.")

    treated_row = wide.loc[treated_unit]
    donor_rows = wide.loc[donor_pool]

    # Balanced-panel check: no missing outcome for treated + donors across the
    # analysis window (pre + post).
    used_times = pre_times + post_times
    block = wide.loc[[treated_unit, *donor_pool], used_times]
    if block.isna().any().any():
        bad = block.stack().isna()
        cells = [(u, t) for (u, t) in bad[bad].index]
        raise ValueError(
            "The panel has missing outcome values for the treated unit or donor "
            f"pool in the analysis window. Offending (unit, time) cells: {cells[:10]}"
            f"{'...' if len(cells) > 10 else ''}."
        )

    # ── construct predictor matrix X (scaled) and outcome matrix Z ──
    if predictors is None:
        predictor_names = [f"{outcome}[t={t}]" for t in pre_times]
        # X = outcome at each pre-treatment period (one predictor per period).
        X1 = np.asarray(treated_row[pre_times].to_numpy(dtype=float), dtype=float)
        X0 = np.asarray(donor_rows[pre_times].to_numpy(dtype=float), dtype=float)
        P = len(pre_times)
    else:
        predictor_names = list(predictors)
        P = len(predictor_names)
        pre_mask = data[time].isin(pre_times)
        agg = (
            data.loc[pre_mask]
            .groupby(entity)[predictors]
            .mean()
            .astype(float)
        )
        if treated_unit not in agg.index:
            raise ValueError(
                f"treated_unit={treated_unit!r} has no pre-treatment predictor "
                "data."
            )
        missing = [d for d in donor_pool if d not in agg.index]
        if missing:
            raise ValueError(
                f"Donor units missing pre-treatment predictor data: {missing}."
            )
        X1 = np.asarray(agg.loc[treated_unit].to_numpy(dtype=float), dtype=float)
        X0 = np.asarray(agg.loc[donor_pool].to_numpy(dtype=float), dtype=float)

    # Z = raw outcome at pre-treatment periods (used for the MSPE loss).
    Z1 = np.asarray(treated_row[pre_times].to_numpy(dtype=float), dtype=float)
    Z0 = np.asarray(donor_rows[pre_times].to_numpy(dtype=float), dtype=float)
    T_pre = len(pre_times)
    N = len(donor_pool)

    # ── standardize each predictor by 1/sd (ddof=1) across all units ─
    X_all = np.vstack([X1, X0])  # (N+1, P)
    sd = X_all.std(axis=0, ddof=1)
    zero_var = np.where(sd == 0.0)[0]
    if zero_var.size > 0:
        bad_names = [predictor_names[j] for j in zero_var]
        raise ValueError(
            "At least one predictor has no variation across the treated unit and "
            f"donor pool (sd = 0). Remove or replace: {bad_names}."
        )
    X1_scaled = X1 / sd
    X0_scaled = X0 / sd

    # ── fixed V (custom.v / predictor_weights) or nested optimization
    if predictor_weights is not None:
        v_raw = np.asarray(
            list(predictor_weights.values())
            if isinstance(predictor_weights, pd.Series)
            else predictor_weights,
            dtype=float,
        )
        if v_raw.shape[0] != P:
            raise ValueError(
                f"predictor_weights has length {v_raw.shape[0]} but there are "
                f"{P} predictors."
            )
        solution_v = np.abs(v_raw) / np.sum(np.abs(v_raw))
        v_success = True
        v_loss = float("nan")
        v_nit = 0
        v_nfev = 0
        v_message = "no V optimization: predictor_weights (custom.v) supplied"
    elif P == 1:
        # Single predictor: R fixes V = 1 and skips the outer loop.
        solution_v = np.array([1.0], dtype=float)
        v_success = True
        v_loss = float("nan")
        v_nit = 0
        v_nfev = 0
        v_message = "no V optimization: single predictor (V = 1)"
    else:
        solution_v, v_success, v_loss, v_nit, v_nfev, v_message = _optimize_v(
            X1_scaled, X0_scaled, Z1, Z0, T_pre, solver_kwargs
        )

    # ── inner W solve at the chosen V ────────────────────────────────
    w_res = _solve_w(X1_scaled, X0_scaled, solution_v, solver_kwargs)
    w = np.asarray(w_res.x, dtype=float)
    w = np.clip(w, 0.0, 1.0)
    w = w / w.sum()

    # ── gap path + MSPE diagnostics ──────────────────────────────────
    synth_all = donor_rows.to_numpy(dtype=float).T @ w  # (n_times,) aligned to wide.columns
    col_pos = {t: i for i, t in enumerate(wide.columns)}
    gap_index = [t for t in used_times if t in col_pos]
    if len(gap_index) == 0:
        raise ValueError(
            "No analysis-window periods overlap the pivoted panel columns; "
            "cannot build a gap path."
        )
    treated_series = np.asarray(
        [float(treated_row[t]) for t in gap_index], dtype=float
    )
    synthetic_series = np.asarray(
        [float(synth_all[col_pos[t]]) for t in gap_index], dtype=float
    )
    gap_series = treated_series - synthetic_series

    gap_path = pd.DataFrame(
        {
            "treated": treated_series,
            "synthetic": synthetic_series,
            "gap": gap_series,
        },
        index=pd.Index(gap_index, name=time),
    )

    pre_mspe = float(np.sum((Z1 - Z0.T @ w) ** 2) / T_pre)
    if post_times:
        Z1_post = np.asarray(treated_row[post_times].to_numpy(dtype=float), dtype=float)
        Z0_post = np.asarray(donor_rows[post_times].to_numpy(dtype=float), dtype=float)
        post_mspe = float(np.sum((Z1_post - Z0_post.T @ w) ** 2) / len(post_times))
    else:
        post_mspe = float("nan")

    weights = pd.Series(w, index=pd.Index(donor_pool, name=entity), name="weight")
    predictor_weights_series = pd.Series(
        solution_v, index=pd.Index(predictor_names, name="predictor"), name="v"
    )

    loss_w = float(
        (X1_scaled - X0_scaled.T @ w)
        @ np.diag(solution_v)
        @ (X1_scaled - X0_scaled.T @ w)
    )

    call = _capture_call(
        data_shape=list(data.shape),
        columns=list(data.columns),
        outcome=outcome,
        treated_unit=treated_unit,
        donor_pool=list(donor_pool),
        entity=entity,
        time=time,
        pre_period=pre_period,
        post_period=post_period,
        predictors=predictor_names,
        predictor_weights=(
            None if predictor_weights is None else list(np.asarray(solution_v))
        ),
        model_type="synth",
        **solver_kwargs,
    )

    formula = (
        f"{outcome} ~ synth({entity}={treated_unit}, "
        f"n_donors={N}, predictors={P})"
    )

    return SynthResult(
        formula=formula,
        outcome=outcome,
        treated_unit=treated_unit,
        donor_pool=list(donor_pool),
        entity=entity,
        time=time,
        pre_period=pre_period,
        post_period=post_period,
        predictors=None if predictors is None else list(predictors),
        weights=weights,
        predictor_weights=predictor_weights_series,
        predictor_names=predictor_names,
        pre_mspe=pre_mspe,
        post_mspe=post_mspe,
        gap_path=gap_path,
        n_donors=N,
        n_pre_periods=T_pre,
        n_post_periods=len(post_times),
        v_success=v_success,
        v_loss=v_loss,
        v_nit=v_nit,
        v_nfev=v_nfev,
        v_message=v_message,
        w_success=bool(w_res.success),
        w_loss=loss_w,
        w_nit=int(w_res.nit),
        w_nfev=int(w_res.nfev),
        w_message=str(w_res.message),
        call=call,
    )


def _solve_w(
    X1_scaled: np.ndarray,
    X0_scaled: np.ndarray,
    v: np.ndarray,
    solver_kwargs: dict[str, Any],
) -> Any:
    """Inner donor-weight QP: minimize (X1 - X0' W)' V (X1 - X0' W), W>=0, sum W=1.

    Returns the raw :class:`scipy.optimize.OptimizeResult`.
    """
    v = np.abs(v)
    v = v / v.sum()
    V = np.diag(v)
    H = X0_scaled @ V @ X0_scaled.T
    N = X0_scaled.shape[0]
    P = X0_scaled.shape[1]
    # L2 regularization for rank-deficient inner QP (donors > predictors).
    # The unregularized Hessian is PSD with rank at most P; adding a tiny ridge
    # makes it strictly convex so the minimizer W is unique and numerically
    # deterministic across BLAS backends / operating systems.
    if N > P:
        H = H + 1e-12 * np.eye(N)
    c = -(X1_scaled @ V @ X0_scaled.T)

    def _obj(w: np.ndarray) -> float:
        return 0.5 * float(w @ H @ w) + float(c @ w)

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0)] * N
    w0 = np.full(N, 1.0 / N)
    return minimize(
        _obj, w0, method="SLSQP", bounds=bounds,
        constraints=constraints, **solver_kwargs,
    )


def _fn_v(
    v: np.ndarray,
    X1_scaled: np.ndarray,
    X0_scaled: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    T_pre: int,
    solver_kwargs: dict[str, Any],
) -> float:
    """Outer objective: pre-treatment outcome MSPE = mean((Z1 - Z0' W)^2)."""
    w_res = _solve_w(X1_scaled, X0_scaled, v, solver_kwargs)
    w = np.asarray(w_res.x, dtype=float)
    err = Z1 - Z0.T @ w
    return float(err @ err) / T_pre


def _optimize_v(
    X1_scaled: np.ndarray,
    X0_scaled: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    T_pre: int,
    solver_kwargs: dict[str, Any],
) -> tuple[np.ndarray, bool, float, int, int, str]:
    """Outer V optimization mirroring R Synth's multi-method procedure.

    R Synth uses ``optimx(method=c("Nelder-Mead", "BFGS"))`` at each of two
    starts (equal-weight and regression-derived), then picks the best result
    across methods via ``collect.optimx``.  We mirror this by running both
    Nelder-Mead (derivative-free, BLAS-insensitive) and SLSQP at each start
    and selecting the result with the lowest objective value.

    Nelder-Mead is unconstrained — ``_fn_v`` → ``_solve_w`` normalises ``|v|``
    internally — so it can explore the full real line, which makes it robust to
    BLAS differences that affect gradient-based methods differently.

    Start 1: equal weights ``1/P``.  Start 2: regression-derived
    ``Beta = (Xall' Xall)^-1 Xall' Zall``, ``V = Beta[-1,] %*% t(Beta[-1,])``,
    ``SV2 = diag(V) / sum(diag(V))``.  Returns the normalized ``abs(par)``
    solution and convergence diagnostics from the winning start.
    """
    P = X1_scaled.shape[0]
    N = X0_scaled.shape[0]
    fn_args = (X1_scaled, X0_scaled, Z1, Z0, T_pre, solver_kwargs)

    constraints = [{"type": "eq", "fun": lambda v: float(np.sum(v) - 1.0)}]
    bounds = [(0.0, 1.0)] * P

    # Nelder-Mead options: honour user maxiter if given, else default.
    nm_opts: dict[str, Any] = {}
    if "maxiter" in solver_kwargs:
        nm_opts["maxiter"] = solver_kwargs["maxiter"]

    def _run_multi_method(start: np.ndarray) -> "OptimizeResult":
        """Run NM + SLSQP from *start*, return the better result.

        NM is derivative-free and BLAS-insensitive, so it is useful as a
        cross-check against gradient-based SLSQP.  However, NM is
        unconstrained (``_fn_v`` normalises |v| internally) and can converge
        to boundary V where one predictor dominates — this produces low
        ``fn_v`` but poor *covariate* balance.  We therefore only accept NM
        when it finds an interior V (no component > 0.95 or < 0.05 after
        normalisation) *and* its objective is strictly better than SLSQP's.
        """
        res_slsp = minimize(
            _fn_v, start, args=fn_args,
            method="SLSQP", bounds=bounds, constraints=constraints, **solver_kwargs,
        )
        res_nm = minimize(
            _fn_v, start, args=fn_args, method="Nelder-Mead", options=nm_opts,
        )
        # Normalise NM result to check for boundary.
        nm_v = np.abs(res_nm.x)
        nm_v = nm_v / nm_v.sum() if nm_v.sum() > 0 else start
        nm_interior = np.all(nm_v > 0.05) and np.all(nm_v < 0.95)
        if nm_interior and res_nm.fun < res_slsp.fun:
            return res_nm
        return res_slsp

    # ── Start 1: equal weights ────────────────────────────────────
    sv1 = np.full(P, 1.0 / P)
    best = _run_multi_method(sv1)

    # ── Start 2: regression-derived (mirrors R's SV2) ─────────────
    treated_row_full = np.concatenate([[1.0], X1_scaled])          # (P+1,)
    donor_rows_full = np.column_stack([np.ones(N), X0_scaled])     # (N, P+1)
    X_full = np.vstack([treated_row_full, donor_rows_full])        # (N+1, P+1)
    z_treated_col = Z1.reshape(T_pre, 1)
    z_donor_cols = Z0.T                                            # (T_pre, N)
    Zall = np.hstack([z_treated_col, z_donor_cols])                # (T_pre, N+1)
    try:
        XtX = X_full.T @ X_full
        Beta = np.linalg.pinv(XtX) @ X_full.T @ Zall.T             # (P+1, T_pre)
        Beta_pred = Beta[1:, :]                                    # (P, T_pre)
        Vmat = Beta_pred @ Beta_pred.T                             # (P, P)
        sv2 = np.diag(Vmat)
        sv2 = sv2 / sv2.sum()
        have_sv2 = True
    except np.linalg.LinAlgError:
        have_sv2 = False

    if have_sv2:
        res2 = _run_multi_method(sv2)
        if res2.fun < best.fun:
            best = res2

    solution_v = np.abs(best.x) / np.sum(np.abs(best.x))
    return (
        solution_v,
        bool(best.success),
        float(best.fun),
        int(best.nit),
        int(best.nfev),
        str(best.message),
    )
