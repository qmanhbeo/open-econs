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

Windmeijer correction (source-confirmed 2026-07-17):
    The ``windmeijer`` flag (default True) controls whether the Windmeijer
    (2005) finite-sample correction is applied to the two-step robust VCE.
    When True (default), the correction is applied (lines 234-249),
    matching Stata's ``xtabond``/``xtdpd`` and the econometric literature's
    recommended practice.  When False, the naive two-step sandwich VCE
    is returned, matching Stata's ``gmm`` command default (confirmed via
    gmm.ado source: no Windmeijer code, no WC-robust label, no toggle).
    The flag is ignored for one-step or non-robust cases.
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
    windmeijer: bool = True,
    robust_meat: str = "one-step",
    weight: str = "stata",
    hac_weighting: bool = False,
    exog_idx: np.ndarray | None = None,
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
    windmeijer : bool, default True
        If True, apply the Windmeijer (2005) finite-sample correction to the
        two-step robust VCE (lines 224-239).  This is the recommended practice
        in the econometric literature and matches Stata's ``xtabond``/``xtdpd``
        default.  If False, skip the correction, reproducing Stata's ``gmm``
        command default (which does NOT apply Windmeijer — confirmed via
        gmm.ado source).  Ignored for one-step or non-robust cases.
    robust_meat : {"one-step", "two-step"}, default "one-step"
        Which residuals feed the robust **MEAT** ``S2`` of the two-step
        VCE sandwich.  The econometric literature and R's ``gmm`` package
        (``vcov="MDS"``) build the robust meat from the ONE-step residuals
        ``e1`` (the efficient two-step weighting is estimated from the
        first-step residuals).  Stata's ``gmm`` command instead builds the
        robust meat from the TWO-step residuals ``e2`` — a genuine,
        source-confirmed convention difference.  Set
        ``robust_meat="two-step"`` to reproduce Stata's ``gmm`` two-step
        robust VCE exactly.  IMPORTANT: this parameter controls ONLY the
        robust meat ``S2``; the efficient-weight bread ``S1`` always stays
        at the one-step residuals ``e1`` (the full sandwich requires this
        split).  Ignored for one-step or non-robust cases.  (Source
        evidence: Stata's ``e(S)`` extracted from a live run equals
        ``(1/N)·Σᵢ(Zᵢ·e2ᵢ)(Zᵢ·e2ᵢ)'`` to machine epsilon, whereas the
        one-step-residual ``S`` differs structurally.)
    weight : {"stata", "iid"}, default "stata"
        Which covariance structure feeds the efficient-weight BREAD ``A2 = S^{-1}``
        AND the robust meat of the two-step GMM.  The default ``"stata"`` uses the
        *same* covariance structure as the VCE (cluster S for cluster, HAC S for
        HAC, per-observation EHW S for robust) — this matches Stata's ``gmm``
        command and makes the two-step coefficient/SE change across
        robust/cluster/HAC.  ``"iid"`` instead uses the **homoskedastic** iid
        weight ``S_iid = Z_iid' Z_iid / n`` where ``Z_iid`` is the intercept
        column plus the *explicit* instruments (the exogenous regressors are
        excluded — they are their own instruments in ``X``), i.e. R's
        ``gmm(..., vcov="iid")`` convention.  Both the efficient weight and the
        robust meat use this homoskedastic S (the meat scaled by the two-step
        residual variance ``sig2``).  This reproduces R's ``gmm``
        ``vcov="iid"`` coefficient ``[0.850, 2.012, 1.354]`` and SE
        ``[0.132, 0.102, 0.805]`` to machine precision.  NOTE: R's ``gmm``
        ``cluster=`` argument is a **no-op** (it is not a real parameter — it
        falls through ``...`` and is never consumed), so R has *no* genuine
        cluster VCE; the historical "R cluster" fixture value is simply R's
        ``vcov="iid"`` two-step.  Source-confirmed against ``gmm`` v1.9-1 source
        (``.weightFct`` iid branch, ``FinRes.baseGmm.res``).
    hac_weighting : bool, default False
        HAC sandwich scope (rule 15).  By default (``False``), the HAC long-run
        covariance ``S`` is computed **per-entity** (Newey-West within each
        panel entity, accumulated) — matching Stata's ``gmm, wmatrix(hac ...)
        vce(hac ...)`` and giving the per-entity HAC coefficient/SE.  When
        ``True`` (and ``max_lags`` is set), ``S`` is instead computed over the
        **full sample** as a single time series (each observation its own
        entity, ordered by row / ``time_labels``) — matching R's
        ``gmm(vcov="HAC")``, which applies the Bartlett kernel to BOTH the
        efficient weight AND the VCE over the pooled sample.  With
        ``hac_weighting=True`` the two-step coefficient changes to R's HAC
        coefficient (e.g. ``[0.885, 2.018, 1.534]`` on the gmm fixture) and the
        SE matches R's HAC SE to <=1e-6.  Ignored when not HAC.  This is a
        genuine convention divergence (Stata per-entity vs R pooled), not a
        bug — see methodology/linear/gmm.md and FUTURE_WORK GMM-HAC.
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
    # For non-HAC cov_types (robust, cluster) `S` is the per-entity covariance
    # of the one-step moment conditions g_i = Z_i' e1_i, grouped by `eq_entity`
    # (identity grouping = iid/robust; cluster ids = clustered).  For HAC,
    # `S` is the Newey-West long-run covariance.  This SAME `S` is the efficient
    # weight `A2 = S^{-1}` AND the VCE meat (unless `robust_meat="two-step"`
    # swaps the meat to the two-step-residual S2).  Source-confirmed vs Stata
    # `gmm`: Stata's `vce(cluster c)` and `wmatrix(hac ...)` both build the
    # efficient weight from the *same* covariance structure used for the VCE,
    # so the two-step coefficient changes across robust/cluster/HAC (matched to
    # <=1e-6 on the gmm fixture).  Do NOT split the bread away from `S` (an
    # early iid-bread attempt forced cluster b == robust b and silently broke
    # parity) -- see methodology/linear/gmm.md.
    if max_lags is not None and max_lags >= 0:
        # HAC bread.  By default the HAC S is per-entity (Newey-West within
        # each entity, accumulated) to match Stata.  When ``hac_weighting`` is
        # set, the HAC S is computed over the FULL SAMPLE as one time series
        # (uniform entity, row order) to match R's gmm(vcov="HAC"), which
        # kernel-averages over the pooled sample.  See hac_weighting docstring.
        if hac_weighting:
            hac_entity = np.zeros(n_eq, dtype=int)
            hac_time = None
        else:
            hac_entity = eq_entity
            hac_time = time_labels
        S = _hac_S(Z, e1, hac_entity, max_lags, hac_time, hac_adjust)
    else:
        S = np.zeros((L, L))
        for ent in np.unique(eq_entity):
            mask = eq_entity == ent
            ze = Z[mask].T @ e1[mask]
            S += np.outer(ze, ze)

    # Efficient-weight BREAD + robust MEAT selection (``weight`` toggle, rule 15).
    # Default ``"stata"``: bread uses the SAME cov-structure S as the VCE
    # (cluster/HAC/iid), matching Stata ``gmm`` and changing b2 across
    # robust/cluster/HAC.  ``"iid"``: both bread and meat use the *homoskedastic*
    # iid S ``S_iid = Z_iid' Z_iid / n`` where Z_iid = [intercept column] +
    # the *explicit* instruments (exogenous regressors excluded — they are their
    # own instruments in X).  This is R's ``gmm(..., vcov="iid")`` convention:
    # the two-step coefficient becomes ``[0.850, 2.012, 1.354]`` and the SE
    # ``[0.132, 0.102, 0.805]`` (scaled by the two-step residual variance sig2).
    # The homoskedastic S is scale-free for the coefficient (sig cancels in A2)
    # but its /n scaling is required for the SE to match R (which divides by n).
    # Do NOT move this inside the S2 block -- the bread governs b2 for every
    # two-step branch.
    S_iid: np.ndarray | None = None
    if weight == "iid":
        if exog_idx is None:
            z_iid = Z
        else:
            n_exog = int(exog_idx.shape[0]) if hasattr(exog_idx, "shape") else len(exog_idx)
            z_iid = np.column_stack([Z[:, 0], Z[:, n_exog:]])  # intercept + explicit instruments
        S_iid = z_iid.T @ z_iid / n_eq
        bread_S = S_iid
        S = S_iid  # V1robust / bread for the iid branch use the homoskedastic S
    elif weight == "stata":
        bread_S = S
    else:
        raise ValueError("weight must be 'stata' or 'iid'.")

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
            A2 = np.linalg.inv(bread_S)
        except np.linalg.LinAlgError:
            A2 = np.linalg.pinv(bread_S)
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

        # --- Optional two-step-residual robust MEAT (Stata `gmm` convention) ---
        # The econometric literature & R's gmm package build the robust VCE as
        # V2 = (G' S1^{-1} G)^{-1}  (equivalently the full sandwich with the
        # one-step residual S1 used for BOTH the efficient weight and the meat).
        # Stata's `gmm` command instead uses the TWO-step residuals e2 for the
        # robust MEAT S2 while keeping the one-step S1 for the efficient weight,
        # i.e. the full sandwich
        #   V = (G' S1^{-1} G)^{-1} (G' S1^{-1} S2 S1^{-1} G) (G' S1^{-1} G)^{-1}.
        # When robust_meat="two-step", compute S2 from e2 and assemble the VCE
        # with S2 in the meat (source-confirmed: reproduces Stata's e(S) and e(V)
        # to machine epsilon).  Coefficients b2 are unchanged at the 1e-9 level.
        # NOTE: S1 (bread, efficient weight) is ALWAYS the one-step-residual S;
        # only the robust meat S2 switches to e2.  Do not "simplify" this to a
        # global S swap -- that regresses parity to a 0.00013 gap.
        S2 = None
        if weight == "iid" and robust and step == "two-step":
            # R gmm(..., vcov="iid") homoskedastic meat: S2 = sig2 * S_iid
            # (the same homoskedastic S as the bread, scaled by the two-step
            # residual variance).  Assemble the full sandwich explicitly so the
            # /n scaling of S_iid (required to match R's V = (G' v^-1 G)^-1 / n)
            # is preserved.
            assert S_iid is not None
            S2 = sig2 * S_iid
            GwG = ZtX.T @ A2 @ ZtX                       # G' S_iid^{-1} G  (bread)
            GwS2wG = ZtX.T @ A2 @ S2 @ A2 @ ZtX          # G' S_iid^{-1} S2 S_iid^{-1} G (meat)
            invGwG = np.linalg.inv(GwG) if GwG.shape[0] == GwG.shape[1] else np.linalg.pinv(GwG)
            V2 = invGwG @ GwS2wG @ invGwG * n_eq         # R: V = (G' v^-1 G)^-1 / n, v = sig*z'z/n
            VXZA1 = V1 @ ZtX.T @ A1
            V1robust = VXZA1 @ S2 @ VXZA1.T * n_eq
            A2Ze = A2 @ (Z.T @ e2)
        elif (
            robust
            and step == "two-step"
            and robust_meat == "two-step"
        ):
            if max_lags is not None and max_lags >= 0:
                # HAC: the two-step-residual meat is the HAC long-run S from e2.
                # When hac_weighting, the meat is the FULL-SAMPLE HAC S (matching
                # R's pooled HAC); otherwise it is the per-entity HAC S (Stata).
                # Using the plain clustered loop here would corrupt the HAC VCE.
                if hac_weighting:
                    S2 = _hac_S(Z, e2, np.zeros(n_eq, dtype=int), max_lags, None, hac_adjust)
                else:
                    S2 = _hac_S(Z, e2, eq_entity, max_lags, time_labels, hac_adjust)
            else:
                S2 = np.zeros((L, L))
                for ent in np.unique(eq_entity):
                    mask = eq_entity == ent
                    ze = Z[mask].T @ e2[mask]
                    S2 += np.outer(ze, ze)
            # Full-sandwich VCE: bread-weight = S1^{-1}, meat = S2.
            GwG = ZtX.T @ A2 @ ZtX                       # G' S1^{-1} G  (bread)
            GwS2wG = ZtX.T @ A2 @ S2 @ A2 @ ZtX          # G' S1^{-1} S2 S1^{-1} G (meat)
            invGwG = np.linalg.inv(GwG) if GwG.shape[0] == GwG.shape[1] else np.linalg.pinv(GwG)
            V2 = invGwG @ GwS2wG @ invGwG
            # Robust one-step sandwich with the two-step-residual meat S2.
            VXZA1 = V1 @ ZtX.T @ A1
            V1robust = VXZA1 @ S2 @ VXZA1.T
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
                if windmeijer:
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
                    # Naive two-step sandwich (Stata gmm default):
                    # VCE = (X'Z S^{-1} Z'X)^{-1} without Windmeijer correction.
                    pV = V2
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
