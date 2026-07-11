from itertools import combinations

import numpy as np


def _minik_contribution(X: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Sum of outer products of score sums within each cluster level."""
    B = np.zeros((X.shape[1], X.shape[1]))
    uniq = np.unique(groups)
    for g in uniq:
        s = scores[groups == g].sum(axis=0)
        B += np.outer(s, s)
    return B


def multiway_cluster_cov(
    X: np.ndarray,
    resid: np.ndarray,
    groups: list[np.ndarray],
) -> np.ndarray:
    """Multi-way cluster-robust variance (Cameron, Gelbach & Miller 2011).

    ``groups`` is a list of integer-coded cluster labels (one array per
    dimension).  Implements the degree-of-freedom-free "minik" estimator that
    combines every non-empty intersection of cluster dimensions with the
    inclusion-exclusion sign ``(-1)^{|S|+1}``.  For two dimensions this reduces
    to the familiar ``V_g1 + V_g2 - V_g1∩g2``.

    Returns the (k, k) variance-covariance matrix of the OLS coefficients.
    """
    n, k = X.shape
    scores = X * resid[:, None]  # (n, k) score contributions x_i * e_i
    XtX_inv = np.linalg.inv(X.T @ X)

    dims = len(groups)
    # Precompute the intersection label for every subset so we only do it once.
    B_total = np.zeros((k, k))
    for size in range(1, dims + 1):
        sign = -1 if size % 2 == 0 else 1
        for combo in combinations(range(dims), size):
            inter = np.zeros(n, dtype=np.int64)
            for d in combo:
                if d == combo[0]:
                    inter = groups[d].astype(np.int64)
                else:
                    # combine into a unique composite label
                    inter = inter * (groups[d].max() + 2) + groups[d].astype(np.int64)
            B_total += sign * _minik_contribution(X, scores, inter)
    V = XtX_inv @ B_total @ XtX_inv
    # The minik estimator can yield a non-positive variance for a coefficient
    # in finite samples (e.g. with nearly nested clusters).  Report those as
    # NaN rather than a silently wrong (negative) standard error.
    diag_v = np.diag(V).copy()
    diag_v = np.where(diag_v > 0, diag_v, np.nan)
    np.fill_diagonal(V, diag_v)
    return V


def newey_west_cov(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int,
    time_index: np.ndarray | None = None,
    cluster: np.ndarray | None = None,
    adjust: bool = False,
) -> np.ndarray:
    """Newey-West (1987) HAC variance with a Bartlett kernel.

    Computes the heteroskedasticity- and autocorrelation-robust variance for a
    time-ordered sample using a Bartlett kernel.  If ``cluster`` is given
    (e.g. entity ids), the long-run covariance is first aggregated within
    clusters and only the between-cluster part is used (panel HAC), matching
    Stata's ``newey`` with ``force`` for repeated time ids.

    When ``adjust=True`` the standard Newey-West (1987) variance is multiplied
    by ``N / (N - K)``, the same finite-sample correction Stata applies
    unconditionally.  This matches R's ``sandwich::NeweyWest(..., adjust=TRUE)``
    but has no theoretical justification in the original NW1987 paper — it is
    borrowed from White's HC1 (MacKinnon & White, 1985).

    Parameters
    ----------
    adjust : bool, default False
        Apply the N/(N-K) degrees-of-freedom correction.

    Returns
    -------
    np.ndarray
        The (k, k) variance-covariance matrix of the OLS coefficients.
    """
    n, k = X.shape
    scores = X * resid[:, None]  # (n, k)
    XtX_inv = np.linalg.inv(X.T @ X)

    if cluster is not None:
        # Aggregate scores within each cluster, then compute the HAC of those
        # cluster-level score vectors (so autocorrelation is across clusters).
        uniq = np.unique(cluster)
        agg = np.zeros((len(uniq), k))
        order = np.argsort(uniq)
        umap = {g: i for i, g in enumerate(uniq)}
        cidx = np.array([umap[g] for g in cluster])
        for i in range(k):
            agg[:, i] = np.bincount(cidx, weights=scores[:, i], minlength=len(uniq))
        s = agg
    else:
        s = scores
        if time_index is not None:
            order = np.argsort(time_index)
            s = s[order]

    m = s.shape[0]
    # Long-run covariance with Bartlett weights.
    S0 = s.T @ s
    Sl = np.zeros((k, k))
    for lag in range(1, max_lags + 1):
        w = 1.0 - lag / (max_lags + 1.0)
        Gamma = np.zeros((k, k))
        for t in range(lag, m):
            Gamma += np.outer(s[t], s[t - lag])
        Sl += w * (Gamma + Gamma.T)
    S = S0 + Sl
    V = XtX_inv @ S @ XtX_inv
    if adjust:
        V *= n / (n - k)
    return V


def white_cov(J: np.ndarray, resid: np.ndarray, kind: str = "HC2") -> np.ndarray:
    """Heteroskedasticity-consistent (MacKinnon & White 1985) covariance for a
    nonlinear least-squares problem, reusing the *linear* sandwich machinery.

    For OLS the robust estimators replace the iid variance ``sigma^2 (X'X)^-1``
    with an empirical sandwich built from the score contributions ``x_i e_i``.
    In NLS the design matrix at the optimum is the Jacobian of the mean function
    with respect to the parameters, ``J = d f(x_i, theta) / d theta`` (n by k),
    and the score contributions are ``J_i e_i``.  Substituting ``J`` for the
    linear ``X`` gives the Gauss-Newton heteroskedasticity-robust covariance.

    The scaling formulas below mirror statsmodels' OLS HC0-HC3 estimators exactly
    (verified against ``statsmodels/regression/linear_model.py``
    ``RegressionResults._HCCM`` / ``HC{0,1,2,3}_se``):

    * HC0: ``scale_i = e_i^2``  (White 1980, raw)
    * HC1: ``scale_i = (n / (n - k)) * e_i^2``  (MacKinnon-White finite-sample)
    * HC2: ``scale_i = e_i^2 / (1 - h_ii)``
    * HC3: ``scale_i = e_i^2 / (1 - h_ii)^2``

    where ``h_ii = diag(J (J'J)^-1 J')`` is the leverage of the Jacobian (the
    Gauss-Newton "hat" diagonal).  The covariance is then

        ``V = (J'J)^-1 J' diag(scale) J (J'J)^-1``

    i.e. ``pinv @ diag(scale) @ pinv.T`` with
    ``pinv = (J'J)^-1 J'`` -- the same construction statsmodels uses
    (``_HCCM``), with no extra overall ``sigma^2`` factor.  This is the
    finite-sample-consistent counterpart to the naive iid variance
    ``sigma^2 (J'J)^-1`` (``sigma^2 = SSR / (n - k)``), which ``nls()`` computes
    separately for ``cov_type="nonrobust"``.

    Parameters
    ----------
    J : np.ndarray
        Jacobian of the mean function w.r.t. the parameters evaluated at the
        solution, shape ``(n, k)``.  For NLS this is ``-res.jac`` when the
        Jacobian was estimated numerically, or the analytic ``df/dtheta``.
    resid : np.ndarray
        Residuals ``y - f(x, theta_hat)`` at the solution, shape ``(n,)``.
    kind : {"HC0", "HC1", "HC2", "HC3"}, default "HC2"
        Which MacKinnon-White estimator to use.  ``"HC2"`` matches the library's
        default robust SE (the same default as ``oe.ols``).

    Returns
    -------
    np.ndarray
        The ``(k, k)`` variance-covariance matrix of the NLS parameters.
    """
    J = np.asarray(J, dtype=float)
    resid = np.asarray(resid, dtype=float).ravel()
    if J.ndim != 2:
        raise ValueError("white_cov expects J with shape (n, k).")
    if resid.shape[0] != J.shape[0]:
        raise ValueError("white_cov: resid length must match J's row count.")
    n, k = J.shape
    if kind not in ("HC0", "HC1", "HC2", "HC3"):
        raise ValueError(f"kind must be one of HC0/HC1/HC2/HC3, got {kind!r}.")
    if n <= k:
        raise ValueError("white_cov requires n > k (more observations than parameters).")

    # pinv = (J'J)^-1 J'  -- the pseudoinverse of the Jacobian.
    pinv = np.linalg.pinv(J)  # (k, n)
    # Leverage h_ii = diag(J @ pinv) = diag(J (J'J)^-1 J').
    h = np.einsum("ij,ji->i", J, pinv)  # (n,)
    h = np.clip(h, 0.0, 1.0 - 1e-12)  # guard against numerical > 1

    e2 = resid ** 2
    if kind == "HC0":
        scale = e2
    elif kind == "HC1":
        scale = e2 * (n / (n - k))
    elif kind == "HC2":
        scale = e2 / (1.0 - h)
    else:  # HC3
        scale = e2 / (1.0 - h) ** 2

    # V = pinv @ diag(scale) @ pinv.T  (matches statsmodels' _HCCM).
    V = pinv @ (scale[:, None] * pinv.T)
    return V


def _as_int_labels(arr: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(arr, return_inverse=True)
    return inv
