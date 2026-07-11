"""Coarsened Exact Matching (CEM) for ATT weights.

Validated against Stata's ``cem`` SSC package
(Blackwell, Iacus, King, Porro 2009, Stata Journal 9(4): 524-546).

Default coarsening follows Stata's default: Sturges' rule
applied to each variable.  Pass explicit cutpoints to override.

Reference
---------
Stata source: C:\\ado\\plus\\c\\cem.ado, C:\\ado\\plus\\c\\cem-mata.do
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call


def _sturges_breaks(x: np.ndarray) -> np.ndarray:
    """Sturges-rule breakpoints, matching ``cem-mata.do`` ``sturges()`` + ``rangen()``.

    Returns *k* = ceil(log2(n) + 1) equally-spaced points covering [min, max].
    The caller then obtains *k-1* bins via ``_coarsen``.
    """
    n = len(x)
    k = int(np.ceil(np.log2(n) + 1))
    return np.linspace(x.min(), x.max(), k)


def _coarsen(x: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Stata-style ``cut()``: 1-indexed bin assignment.

    Matches ``cem-mata.do`` ``cut()`` implementation exactly.
    """
    b = (x >= breaks[0]).astype(np.intp)
    for cut_val in breaks[1:]:
        b += (x > cut_val).astype(np.intp)
    return b


def _compute_strata(coarsened_vars: dict[str, np.ndarray]) -> np.ndarray:
    """Assign stratum IDs from coarsened variable levels.

    Each unique combination of coarsened levels → a stratum ID (1-indexed,
    sorted lexicographically, matching Stata's ``uniqrows`` ordering).
    """
    keys = np.array([
        " ".join(str(v) for v in row)
        for row in zip(*coarsened_vars.values())
    ], dtype=str)
    unique_keys = np.unique(keys)
    lookup = {k: i + 1 for i, k in enumerate(unique_keys)}
    return np.array([lookup[k] for k in keys], dtype=np.intp)


def _compute_matcher(strata: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Flag strata that contain *both* treated and control observations."""
    matched = np.zeros(len(strata), dtype=bool)
    for s in np.unique(strata):
        in_s = strata == s
        has_t = (treatment[in_s] == 1).any()
        has_c = (treatment[in_s] == 0).any()
        if has_t and has_c:
            matched[in_s] = True
    return matched


def _compute_weights(
    strata: np.ndarray,
    treatment: np.ndarray,
    matched: np.ndarray,
) -> np.ndarray:
    """ATT weights matching Stata's ``cem-mata.do`` formula (lines 297-301).

    Treated unit:        weight = 1
    Control (matched):   weight = n_T(s) / n_C(s) * N_C_matched / N_T_matched
    Unmatched:           weight = 0

    Verified numerically against Stata output on a hand-checkable dataset
    with 3 strata and unbalanced counts.
    """
    n = len(treatment)
    weights = np.zeros(n, dtype=float)

    unique_strata = np.unique(strata[matched])
    is_treat = treatment == 1
    is_control = treatment == 0
    n_t_matched = int((is_treat & matched).sum())
    n_c_matched = int((is_control & matched).sum())

    per_stratum_base = np.zeros(int(strata.max()) + 1, dtype=float)
    for s in unique_strata:
        in_s = strata == s
        n_t = int((treatment[in_s] == 1).sum())
        n_c = int((treatment[in_s] == 0).sum())
        if n_c > 0 and n_t_matched > 0:
            per_stratum_base[s] = (n_t / n_c) * (n_c_matched / n_t_matched)
        else:
            per_stratum_base[s] = 0.0

    weights = per_stratum_base[strata] * matched.astype(float)

    treated_mask = (treatment == 1) & matched
    weights[treated_mask] = 1.0

    return weights


class CEMResult(BaseModel):
    """Result of a Coarsened Exact Matching procedure.

    Immutable result with ``.tidy()``, ``.summary()``, and ``.export()``.
    """

    def __init__(
        self,
        *,
        original_data: pd.DataFrame,
        strata: np.ndarray,
        weights: np.ndarray,
        matched: np.ndarray,
        treatment: np.ndarray,
        coarsened: dict[str, np.ndarray],
        breakpoints: dict[str, np.ndarray],
        call: dict[str, Any],
    ) -> None:
        self.original_data = original_data
        self._strata = strata
        self._weights = weights
        self._matched = matched
        self._treatment = treatment
        self._coarsened = coarsened
        self._breakpoints = breakpoints
        self.call = call
        self.timestamp = __import__("datetime").datetime.now()
        self.package_version = __version__
        self._freeze()

    @property
    def strata(self) -> pd.Series:
        return pd.Series(self._strata, name="cem_strata", index=self.original_data.index)

    @property
    def weights(self) -> pd.Series:
        return pd.Series(self._weights, name="cem_weights", index=self.original_data.index)

    @property
    def matched(self) -> pd.Series:
        return pd.Series(
            self._matched.astype(int), name="cem_matched",
            index=self.original_data.index,
        )

    @property
    def n_matched_strata(self) -> int:
        return int(np.unique(self._strata[self._matched]).size)

    def tidy(self) -> pd.DataFrame:
        t = self._treatment.astype(bool)
        m = self._matched
        w = self._weights

        n_t_all = int(t.sum())
        n_c_all = int((~t).sum())
        n_t_matched = int((t & m).sum())
        n_c_matched = int(((~t) & m).sum())
        n_strata = int(np.unique(self._strata).size)
        n_mstrata = self.n_matched_strata
        sum_wt = float(w[t].sum())
        sum_wc = float(w[~t].sum())

        return pd.DataFrame({
            "term": [
                "N (all, treated)", "N (all, control)",
                "N (matched, treated)", "N (matched, control)",
                "N (strata)", "N (matched strata)",
                "sum(weights, treated)", "sum(weights, control)",
            ],
            "value": [
                n_t_all, n_c_all,
                n_t_matched, n_c_matched,
                n_strata, n_mstrata,
                sum_wt, sum_wc,
            ],
        })

    def summary(self) -> str:
        t = self._treatment.astype(bool)
        m = self._matched

        n_t_all = int(t.sum())
        n_c_all = int((~t).sum())
        n_t_matched = int((t & m).sum())
        n_c_matched = int(((~t) & m).sum())
        n_strata = int(np.unique(self._strata).size)
        n_mstrata = self.n_matched_strata

        lines = [
            "              Coarsened Exact Matching Results               ",
            "============================================================",
            f"Number of strata:               {n_strata:>8d}",
            f"Number of matched strata:       {n_mstrata:>8d}",
            "",
            f"Treated (all / matched):        {n_t_all:>8d} / {n_t_matched:<d}",
            f"Control (all / matched):        {n_c_all:>8d} / {n_c_matched:<d}",
            "",
            "Breakpoints:",
        ]
        for var, bp in self._breakpoints.items():
            bp_str = "[" + ", ".join(f"{v:.4g}" for v in bp) + "]"
            lines.append(f"  {var:<20s} {bp_str}")
        lines.append("============================================================")
        return "\n".join(lines)

    def _export_core(self) -> dict[str, Any]:
        return {
            "n_strata": int(np.unique(self._strata).size),
            "n_matched_strata": self.n_matched_strata,
            "n_matched": int(self._matched.sum()),
            "breakpoints": {k: v.tolist() for k, v in self._breakpoints.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(self._export_core())
        return d


def cem(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str] | None = None,
    cutpoints: dict[str, int | list[float] | str] | None = None,
) -> CEMResult:
    """Coarsened Exact Matching.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data.
    treatment : str
        Name of the binary treatment column.
    covariates : list of str, optional
        Variables to coarsen and match on.  If omitted, all numeric columns
        other than *treatment* are used.
    cutpoints : dict, optional
        Per-variable coarsening specification.

        * ``int`` — number of equally-spaced bins.
        * ``list[float]`` — explicit breakpoints (used directly with
          ``_coarsen``, matching Stata syntax).
        * ``"sturges"`` — Sturges' rule (default if a variable is absent
          from the dict).
        * ``"exact"`` — no coarsening (match on original values).

        Variables not present in *cutpoints* default to Sturges' rule,
        matching Stata's ``cem`` default.

    Returns
    -------
    CEMResult
        Immutable result with strata, weights, and matched flags.

    Examples
    --------
    >>> import open_econs as oe
    >>> r = oe.cem(df, treatment="t", covariates=["x1", "x2"])
    >>> r.tidy()
    >>> r.summary()
    """
    call = _capture_call(
        treatment=treatment, covariates=covariates, cutpoints=cutpoints,
    )

    _data = data.copy()

    if covariates is None:
        covariates = [
            c for c in _data.columns
            if c != treatment and np.issubdtype(_data[c].dtype, np.number)
        ]
    if not covariates:
        raise ValueError("At least one covariate is required for CEM.")

    t = _data[treatment].values
    cutpoints = cutpoints or {}

    # Build coarsened variables and breakpoints
    coarsened: dict[str, np.ndarray] = {}
    bp_store: dict[str, np.ndarray] = {}

    for var in covariates:
        x = _data[var].values.astype(float)
        spec = cutpoints.get(var, "sturges")

        if isinstance(spec, str):
            if spec == "exact":
                breaks = None
            elif spec == "sturges":
                breaks = _sturges_breaks(x)
            else:
                raise ValueError(
                    f"Unknown binning method '{spec}' for '{var}'. "
                    f"Use 'sturges', 'exact', int, or list[float]."
                )
        elif isinstance(spec, int):
            if spec < 2:
                raise ValueError(f"'{var}': at least 2 bins required, got {spec}.")
            breaks = np.linspace(x.min(), x.max(), spec)
        elif isinstance(spec, (list, tuple, np.ndarray)):
            breaks = np.asarray(spec, dtype=float)
            if breaks.ndim != 1:
                raise ValueError(f"'{var}': breakpoints must be 1-D.")
            if len(breaks) == 0:
                breaks = None
            elif len(breaks) == 1:
                breaks = np.concatenate([[x.min()], breaks])
        else:
            raise TypeError(
                f"'{var}': unsupported type {type(spec).__name__}. "
                f"Use int, list[float], 'sturges', or 'exact'."
            )

        if breaks is None:
            coarsened[var] = x.copy()
        else:
            coarsened[var] = _coarsen(x, breaks).astype(float)
            bp_store[var] = breaks

    strata = _compute_strata(coarsened)
    matched = _compute_matcher(strata, t)
    weights = _compute_weights(strata, t, matched)

    return CEMResult(
        original_data=_data,
        strata=strata,
        weights=weights,
        matched=matched,
        treatment=t,
        coarsened=coarsened,
        breakpoints=bp_store,
        call=call,
    )
