"""Shared internal GMM solver core.

General, estimator-agnostic two-step GMM machinery: one-step / two-step
estimation, per-entity clustered weighting matrix ``S``, one-step and
two-step (Windmeijer 2005) robust standard errors, the small-sample
multiplier, and the Hansen J overidentification test.

This is the engine that powers :func:`open_econs.models.linear.abond.abond`
and is the intended foundation for a future general ``gmm()`` estimator.
It operates purely on the moment/weighting matrices ``Y``, ``X``, ``Z`` and a
grouping key ``eq_entity`` — it carries **no** Arellano-Bond- or
panel-specific knowledge other than what the caller chooses to encode in the
one-step weighting matrix ``W``.

The one-step weighting is ``A1 = (Z' W Z)^{-1}``.  ``W`` defaults to the
identity (the plain/iid case, i.e. ``(Z'Z)^{-1}``).  Arellano-Bond passes the
first-difference block-diagonal operator ``H`` (built from
``_build_h`` in ``abond.py``) as ``W`` so its behavior is preserved exactly.

J-statistic convention (source-confirmed 2026-07-17):
    The one-step non-robust J uses the model-based weighting
    ``A1 = (Z'Z)^{-1} / sig2`` (line 158 below), giving
    ``J = g' (Z'Z)^{-1} g / sig2`` where ``sig2 = e'e / N``.
    This matches R's ``gmm::specTest(tsls)`` (confirmed to machine
    epsilon on the shared fixture).  Stata's ``e(J) = Q * N`` uses the
    robust sandwich ``S = (1/N) Σ g_i g_i'`` instead, giving a
    numerically different J (~4.03 vs ~3.77 on the fixture).  Both are
    valid; the model-based variant is the one consistent with the
    textbook GMM J test under homoskedasticity (Hansen 1982).  The
    two-step J always uses the efficient weighting ``A2 = S^{-1}``
    regardless of the robust flag.
"""

from typing import Any

import numpy as np
from scipy.stats import chi2 as _chi2


def _hac_S(
    Z: np.ndarray,
    e: np.ndarray,
    eq_entity: np.ndarray,
    max_lags: int,
    time_labels: np.ndarray | None,
    adjust: bool,
) -> np.ndarray:
    L = Z.shape[1]
    n_tot = Z.shape[0]
    S = np.zeros((L, L))
    for ent in np.unique(eq_entity):
        mask = eq_entity == ent
        Z_e = Z[mask]
        e_e = e[mask]
        if time_labels is not None:
            order = np.argsort(time_labels[mask], kind="stable")
            Z_e = Z_e[order]
            e_e = e_e[order]
        moments = Z_e * e_e[:, None]
        T_ent = moments.shape[0]
        if T_ent == 0:
            continue
        S_ent = moments.T @ moments
        for lag in range(1, min(max_lags, T_ent - 1) + 1):
            w = 1.0 - lag / (max_lags + 1.0)
            Gamma = np.zeros((L, L))
            for t in range(lag, T_ent):
                Gamma += np.outer(moments[t], moments[t - lag])
            S_ent += w * (Gamma + Gamma.T)
        S += S_ent
    if adjust:
        S *= n_tot / max(float(n_tot - Z.shape[1]), 1.0)
    return S


def estimate_gmm(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    eq_entity: np.ndarray,
    step: str,
    robust: bool = False,
    W: np.ndarray | None = None,
    sig2_scale: float = 1.0,
    small_sample_correction: bool = False,
    time_labels: np.ndarray | None = None,
    max_lags: int | None = None,
    hac_adjust: bool = False,
) -> dict[str, Any]:
    """General two-step GMM estimator, mirroring xtabond2's Mata source (v3.7.2).

    Implements one-step / two-step / robust / Windmeijer-corrected VCEs exactly
    as the Mata code assembles them, for *arbitrary* moment conditions and
    instrument matrices ``Y, X, Z`` clustered by ``eq_entity``.  Key
    conventions:

    * One-step weighting ``A1 = (Z'WZ)^{-1}``.  ``W`` is supplied by the caller
      (default identity: the plain/iid ``(Z'Z)^{-1}``).  Arellano-Bond supplies
      the first-difference block-diagonal operator ``H`` so its weighting is
      reproduced exactly.
    * Two-step weighting ``A2 = S^{-1}`` with ``S = Σᵢ (Zᵢ'e1ᵢ)(Zᵢ'e1ᵢ)'`` built
      per-entity from the ONE-step residuals ``e1`` (Mata 450-460).
    * Same ``S`` is reused for ``A2`` (two-step weighting) and for the one-step
      robust sandwich ``V1robust`` (Mata 462-464).
    * Windmeijer two-step-robust VCE uses ``V2`` in two terms and the one-step
      robust ``V1robust`` in the ``D*V1robust*D'`` term (Mata 522-523).
    * Small-sample multiplier (Mata 562-565): one-step non-robust uses
      ``wttot/(wttot-k)``; all other cases use
      ``(NObs-1)/(NObs-k)·NGroups/(NGroups-1)``.
    * ``m2VZXA = -2·V·(ZX'·A)`` is built from the *pre*-small-sample V/A
      (Mata 549), while the V/sig2 fed into the AR test are the *post*-
      correction values (Mata 614) — operand timing must be preserved.

    Parameters
    ----------
    Y : ndarray, shape (n_eq,)
        Moment LHS (the dependent vector, after any model transform).
    X : ndarray, shape (n_eq, p)
        Regressor matrix.
    Z : ndarray, shape (n_eq, L)
        Instrument / moment matrix.
    eq_entity : ndarray, shape (n_eq,)
        Grouping key (e.g. panel entity) used to build the per-group ``S``
        and for the group counts in the small-sample multiplier.
    step : {"one-step", "two-step"}
        GMM step.
    robust : bool, default False
        Use cluster-robust sandwich standard errors.
    W : ndarray, shape (n_eq, n_eq), optional
        One-step weighting matrix.  Defaults to identity (plain/iid).
    sig2_scale : float, default 1.0
        Scale applied to the residual variance ``sig2 = sig2_scale · e'e / n``.
        The default ``1.0`` is the generic GMM convention.  Arellano-Bond passes
        ``0.5`` (its first-difference ``1/2`` normalization: ``sig2 = e'e / (2n)``).
    small_sample_correction : bool, default False
        If True, apply the estimator-specific finite-sample multipliers to the
        variance ``V`` (one-step non-robust: ``wttot/(wttot-k)``; otherwise
        ``(NObs-1)/(NObs-k)·NGroups/(NGroups-1)``) and the Stata/AB
        ``wttot/(wttot-k)`` multiplier to ``sig2``, mirroring xtabond2's Mata
        source.  If False (generic default), no small-sample correction is
        applied to ``V`` or ``sig2``.
    """
    n_eq = Y.shape[0]
    N = float(len(np.unique(eq_entity)))
    p = X.shape[1]
    L = Z.shape[1]

    ZtX = Z.T @ X  # (L, p)  == ZX'
    ZtY = Z.T @ Y  # (L,)

    # --- One-step weighting A1 = (Z'WZ)^{-1} ---
    W_inner = np.eye(n_eq) if W is None else W
    ZtHZ = Z.T @ W_inner @ Z
    try:
        A1_raw = np.linalg.inv(ZtHZ)
    except np.linalg.LinAlgError:
        A1_raw = np.linalg.pinv(ZtHZ)
    G1 = ZtX.T @ A1_raw @ ZtX
    try:
        V1_raw = np.linalg.inv(G1)
    except np.linalg.LinAlgError:
        V1_raw = np.linalg.pinv(G1)
    b1 = V1_raw @ (ZtX.T @ A1_raw @ ZtY)
    e1 = Y - X @ b1

    wttot = float(n_eq)        # = NObs (no weights, difference GMM)
    NGroups = float(N)
    k = float(p)
    sig2 = sig2_scale * float(e1 @ e1) / wttot
    # Model-based J: A1 = (Z'Z)^{-1} / sig2 gives J = g'(Z'Z)^{-1}g / sig2.
    # This matches R's specTest(tsls).  Stata's e(J) uses robust S instead
    # (see module docstring).  Kept as-is per source-confirmed convention audit.
    A1 = A1_raw / sig2
    V1 = V1_raw * sig2

    onestepnonrobust = (step == "one-step") and (not robust)

    # --- Per-entity S from ONE-step residuals (A2, V1robust, Hansen) ---
    if max_lags is not None and max_lags >= 0:
        S = _hac_S(Z, e1, eq_entity, max_lags, time_labels, hac_adjust)
    else:
        S = np.zeros((L, L))
        for ent in np.unique(eq_entity):
            mask = eq_entity == ent
            ze = Z[mask].T @ e1[mask]
            S += np.outer(ze, ze)

    if onestepnonrobust:
        b = b1
        pV_pre = V1
        pA_pre: np.ndarray = A1
        pe = e1
        pV = V1
    else:
        if robust:
            VXZA1 = V1 @ ZtX.T @ A1             # V1 * (ZX' A1)
            V1robust = VXZA1 @ S @ VXZA1.T
        try:
            A2 = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            A2 = np.linalg.pinv(S)
        G2 = ZtX.T @ A2 @ ZtX
        try:
            V2 = np.linalg.inv(G2)
        except np.linalg.LinAlgError:
            V2 = np.linalg.pinv(G2)
        b2 = V2 @ (ZtX.T @ A2 @ ZtY)
        e2 = Y - X @ b2
        if step == "two-step":
            sig2 = sig2_scale * float(e2 @ e2) / wttot   # Mata 480: two-step sig2 from e2
        A2Ze = A2 @ (Z.T @ e2)
        if step == "one-step":
            b = b1
            pV_pre = V1
            pA_pre = A1
            pe = e1
            pV = V1robust if robust else V1
        else:  # two-step
            b = b2
            pV_pre = V2
            pA_pre = A2
            pe = e2
            if robust:
                # Windmeijer (2005) correction — Mata 510-523
                VXZA2 = V2 @ ZtX.T @ A2
                D = np.zeros((L, p))
                for ent in np.unique(eq_entity):
                    mask = eq_entity == ent
                    ze = Z[mask].T @ e1[mask]           # (L,)  Z_i' e1_i (one-step)
                    ZXi = Z[mask].T @ X[mask]           # (L, p)
                    # Mata 518: term1 = scalar (Z_i'e1_i · A2Ze) * ZXi ;
                    # term2 = outer(Z_i'e1_i, A2Ze'·ZXi) — per-row-varying scale.
                    s1 = ze @ A2Ze                      # scalar
                    term1 = s1 * ZXi                    # (L, p)
                    term2 = np.outer(ze, A2Ze @ ZXi)    # (L, p)
                    D += term1 + term2
                D = VXZA2 @ D                           # -> (p, p)
                V2robust = V2 + D @ V1robust @ D.T + 2.0 * D @ V2
                pV = V2robust
            else:
                pV = V2

    # --- Small-sample correction (Mata 562-565) ---
    # NB: Mata multiplies V by the *branch-specific* multiplier but multiplies
    # sig2 by `tmp = wttot/(wttot-k)` ALWAYS (line 565 uses `tmp`, not the
    # branch multiplier).  Keep the two multipliers separate.  Both are applied
    # only when `small_sample_correction` is requested (generic default: none).
    if small_sample_correction:
        if onestepnonrobust:
            small_mult = wttot / (wttot - k)
        else:
            small_mult = ((wttot - 1.0) / (wttot - k)) * (NGroups / (NGroups - 1.0))
        sig2_mult = wttot / (wttot - k)
    else:
        small_mult = 1.0
        sig2_mult = 1.0
    V = pV * small_mult
    sig2 = sig2 * sig2_mult

    se = np.sqrt(np.maximum(np.diag(V), 0.0))

    # --- Hansen J (A1 for one-step, A2 for two-step) ---
    g_all = Z.T @ pe
    A_used = A1 if onestepnonrobust else A2
    dof_j = L - p
    J = float(g_all @ A_used @ g_all)
    p_j = float(1.0 - _chi2.cdf(J, dof_j)) if dof_j > 0 else float("nan")

    # --- Windmeijer/AR prep: m2VZXA from PRE-small V/A (Mata 549) ---
    m2VZXA = -2.0 * pV_pre @ (ZtX.T @ pA_pre)

    # pV_ar is the pre-small V passed to _ARTests (Stata line 614).
    # The reporting V and sig2 get the small multiplier; m2VZXA and pV_ar
    # use the base (uncorrected) values.
    return {
        "b": b, "se": se, "e": pe, "Z": Z, "X": X, "Y": Y,
        "sig2": sig2, "V": V,
        "J": J, "dof_j": dof_j, "p_j": p_j,
        "m2VZXA": m2VZXA, "pV": V, "pV_ar": pV,
        "eq_entity": eq_entity, "n_eq": n_eq, "N": N, "p": p, "L": L,
        "step": step, "robust": robust,
    }
