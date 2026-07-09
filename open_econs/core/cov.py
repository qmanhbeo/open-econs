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
) -> np.ndarray:
    """Newey-West (1987) HAC variance with a Bartlett kernel.

    Computes the heteroskedasticity- and autocorrelation-robust variance for a
    time-ordered sample.  If ``cluster`` is given (e.g. entity ids), the
    long-run covariance is first aggregated within clusters and only the
    between-cluster part is used (panel HAC), matching Stata's ``newey`` with
    ``force`` for repeated time ids.

    Returns the (k, k) variance-covariance matrix of the OLS coefficients.
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
    return V


def _as_int_labels(arr: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(arr, return_inverse=True)
    return inv
