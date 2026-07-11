"""Rosenbaum bounds sensitivity analysis for matched pairs.

Implements the Wilcoxon signed-rank-based Rosenbaum bounds (Rosenbaum 2002,
ch. 4, Eq. 4.9, p. 112) for sensitivity to hidden bias in matched-pair
observational studies.

Reference implementations
-------------------------
- Stata: rbounds (Markus Gangl, v1.1.6, SSC), source read in full.
- R: rbounds (Luke J. Keele, v2.2, CRAN), source read in full.

Validated primarily against Stata (project convention).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm
from scipy.stats import rankdata as _rankdata

from open_econs._version import __version__
from open_econs.core.base import BaseModel


class RosenbaumBoundsResult(BaseModel):
    """Result of a Rosenbaum bounds sensitivity analysis.

    Immutable result with ``.tidy()`` (DataFrame of Γ, lower/upper p-values),
    ``.summary()`` (text), ``.critical_gamma`` (smallest Γ where the upper
    bound exceeds 0.05), and the standard ``.export()`` / ``.to_latex()`` /
    ``.to_html()`` interface.
    """

    def __init__(
        self,
        *,
        gamma: np.ndarray,
        p_upper: np.ndarray,
        p_lower: np.ndarray,
        n_pairs: int,
        call: dict[str, Any],
    ) -> None:
        self._gamma = gamma
        self._p_upper = p_upper
        self._p_lower = p_lower
        self._n_pairs = n_pairs
        self.call = call
        self.timestamp = __import__("datetime").datetime.now()
        self.package_version = __version__
        self._freeze()

    @property
    def critical_gamma(self) -> float | None:
        """Smallest Γ where the upper-bound p-value exceeds 0.05.

        If the upper bound stays at or below 0.05 for all computed gamma
        values, returns None (the finding is robust to hidden bias at all
        tested levels of Γ).

        Higher values indicate a more robust finding.  A value of e.g. 2.5
        means the result remains significant at the 5 % level until Γ
        exceeds 2.5.
        """
        mask = self._p_upper > 0.05
        if not mask.any():
            return None
        return float(self._gamma[mask][0])

    def tidy(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Gamma": self._gamma,
            "lower_bound_p": self._p_lower,
            "upper_bound_p": self._p_upper,
        })

    def summary(self) -> str:
        cg = self.critical_gamma
        cg_str = f"{cg:.3f}" if cg is not None else "None (robust at all Γ)"
        header = (
            f"              Rosenbaum Bounds Sensitivity Analysis          \n"
            f"================================================================\n"
            f"Matched pairs:                 {self._n_pairs}\n"
            f"Critical Gamma (p_up > 0.05):  {cg_str}\n"
            f"================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n================================================================\n"


def rosenbaum_bounds(
    d: np.ndarray | pd.Series | list[float],
    gamma_max: float = 6.0,
    gamma_inc: float = 0.1,
) -> RosenbaumBoundsResult:
    """Rosenbaum bounds sensitivity analysis on matched-pair differences.

    Implements the Wilcoxon signed-rank Rosenbaum bounds (Rosenbaum 2002,
    ch. 4, Eq. 4.9, p. 112).  Zero-difference handling follows **Stata's**
    ``rbounds`` (``rbrksm`` lines 131-132) convention: pairs with ``d_i = 0``
    are included in the rank computation (occupying the lowest rank positions)
    but contribute zero to the test statistic, expectations, and variance
    (``psp = 0, psm = 0``).  This differs from R's ``rbounds`` package which
    drops zero-difference pairs before ranking.

    Tie-handling follows **Stata 17's** ``egen rank()`` convention (average
    rank for tied absolute differences), verified empirically against Stata
    output.  ``scipy.stats.rankdata(method="average")`` is used.

    Parameters
    ----------
    d : array-like
        Vector of within-pair differences d_i = Y_{Ti} - Y_{Ci} for each
        matched pair.  Each element corresponds to exactly one pair.
    gamma_max : float, default 6.0
        Maximum value of the sensitivity parameter Γ.  The grid ranges
        from 1.0 to ``gamma_max`` inclusive, in steps of ``gamma_inc``.
    gamma_inc : float, default 0.1
        Step size for the Γ grid.

    Returns
    -------
    RosenbaumBoundsResult
    """
    d_arr = np.asarray(d, dtype=float)
    d_arr = d_arr[~np.isnan(d_arr)]
    if len(d_arr) < 2:
        raise ValueError(
            "At least 2 non-NaN pairs are required for Rosenbaum bounds."
        )

    n = len(d_arr)
    abs_d = np.abs(d_arr)
    ranks = _rankdata(abs_d, method="average")

    is_pos = d_arr > 0.0
    T = float(np.sum(ranks[is_pos]))

    gamma_grid = np.arange(1.0, gamma_max + 1e-12, gamma_inc)
    p_up = np.empty_like(gamma_grid)
    p_low = np.empty_like(gamma_grid)

    is_zero = d_arr == 0.0

    for idx, g in enumerate(gamma_grid):
        if is_zero.any():
            psp = np.where(is_zero, 0.0, g / (1.0 + g))
            psm = np.where(is_zero, 0.0, 1.0 / (1.0 + g))
        else:
            psp = np.full(n, g / (1.0 + g))
            psm = np.full(n, 1.0 / (1.0 + g))

        et_plus = float(np.sum(ranks * psp))
        et_minus = float(np.sum(ranks * psm))
        vt = float(np.sum(ranks ** 2 * psp * (1.0 - psp)))

        if vt <= 0.0:
            p_up[idx] = 1.0
            p_low[idx] = 1.0
            continue

        sd = np.sqrt(vt)
        z_plus = (T - et_plus) / sd
        z_minus = (T - et_minus) / sd

        p_up[idx] = 1.0 - _norm.cdf(z_plus)
        p_low[idx] = 1.0 - _norm.cdf(z_minus)

    return RosenbaumBoundsResult(
        gamma=gamma_grid,
        p_upper=p_up,
        p_lower=p_low,
        n_pairs=n,
        call={"gamma_max": gamma_max, "gamma_inc": gamma_inc},
    )
