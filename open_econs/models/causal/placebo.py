"""Synthetic control placebo inference (Abadie-Diamond-Hainmueller).

Placebo-in-space and placebo-in-time permutation inference built on top of the
already-validated :func:`synth` point estimator.  Each placebo re-runs the same
core ``synth()`` solver once per placebo unit (space) or once per candidate
pre-treatment date (time); no estimator logic is duplicated.

Inference follows the ADH permutation convention:

* For each placebo, the statistic is the ratio ``post_MSPE / pre_MSPE``.
* The reported p-value is the **fraction** of placebo ratios that are ``>=``
  the treated unit's own ratio.  This is the ADH permutation / "rank" p-value:
  Abadie, Diamond & Hainmueller (2010), *Synthetic Control Methods for
  Comparative Case Studies: Estimating the Effect of California's Tobacco
  Control Program*, Journal of the American Statistical Association, 105(490),
  493-505, §3.2 (and the empirical application, where the reported p-value is
  "the fraction of control states whose post/pre-treatment RMSPE ratio is at
  least as large as the treated unit's"); Abadie, Diamond & Hainmueller (2015),
  *Comparative Politics and the Synthetic Control Method*, American Journal of
  Political Science, 59(2), 495-510, §3 (the "in-space" and "in-time" placebo
  tests are described there as exactly this proportion).
* ADH also discuss dropping placebo units whose *pre*-treatment fit is so poor
  that they are uninformative: a placebo whose pre-treatment RMSPE is far larger
  than the treated unit's is not a meaningful counterfactual (ADH 2010, §3.4 /
  application; 2015, §3).  We expose this as the optional, **space-only**
  ``exclude_pre_mspe_multiple`` parameter (a multiple of the treated unit's
  pre-treatment MSPE).  The default is ``None`` (no exclusion) so the behavior
  is never applied silently.  ADH's *applications* use different cutoffs for
  different studies (e.g. the German-reunification study, ADH 2015, uses a 10x
  cut on pre-treatment RMSPE), but there is **no single canonical multiplier**:
  the literature's chosen thresholds vary by application, so this package
  deliberately does not hard-code one as a silent default.  The parameter is
  space-only; ``placebo_time`` does not accept it (the "pre-MSPE" in the
  in-time loop is the treated unit's own fit against itself at different
  candidate cutoffs, not a fit-quality comparison across independent placebos,
  so the exclusion concept does not carry over).

This is new, additive code only: no existing estimator is modified.
"""

from __future__ import annotations

import concurrent.futures as _cf
from typing import Any, Optional, Sequence

import pandas as pd

from open_econs._version import __version__
from open_econs.core.base import BaseModel
from open_econs.core.call_capture import capture_call as _capture_call
from open_econs.core.results import SynthResult


def _adh_permutation_p_value(ratios: pd.Series, treated_ratio: float) -> float:
    """Fraction of placebo ratios ``>=`` the treated unit's ratio (ADH)."""
    if len(ratios) == 0:
        return float("nan")
    return float((ratios >= treated_ratio).mean())


# Below which a permutation loop is run sequentially.  Each ``synth()`` fit costs
# ~1s; the process pool pays a fixed spawn + per-item pickling overhead (the
# DataFrame and the full ``open_econs`` import must be re-created in every
# worker), so for small loops parallelising is a net loss.  Benchmarks on this
# machine: N=6 -> 0.90x (slower), N=12 -> 1.68x (faster).  The threshold is
# deliberately conservative so default behaviour never regresses.
_MIN_PARALLEL_ITEMS = 8


def _one_synth(spec: dict[str, Any]) -> dict[str, Any]:
    """Run a single ``synth`` fit from a picklable spec.

    Pure worker for the permutation loop: takes the exact kwargs a
    ``synth`` fit needs (data is passed by the caller), returns the three
    quantities the placebo loop consumes plus any exception.  Kept at module
    level so it is picklable across ``ProcessPoolExecutor`` (Windows spawn).
    """
    from open_econs.models.causal.synth import synth

    key = spec.pop("__key__")
    try:
        r = synth(**spec)
        return {
            "key": key,
            "ok": True,
            "pre_mspe": float(r.pre_mspe),
            "post_mspe": float(r.post_mspe),
            "gap": r.gap_path["gap"],
        }
    except Exception as exc:  # noqa: BLE001 - mirrored from the sequential loop
        return {"key": key, "ok": False, "error": str(exc)}


def _parmap(
    specs: Sequence[dict[str, Any]],
    parallel: bool,
) -> list[dict[str, Any]]:
    """Run permutation ``synth`` fits, optionally across a process pool.

    Sequential and parallel paths are numerically identical: ``synth`` is a pure
    function (no shared mutable state; its only RNG use is a fixed-seed Dirichlet
    start in the inner QP), so each worker reproduces the sequential result
    exactly.  Returns the list of per-item result dicts (order preserved).
    """
    if parallel and len(specs) >= _MIN_PARALLEL_ITEMS:
        with _cf.ProcessPoolExecutor() as ex:
            return list(ex.map(_one_synth, list(specs)))
    return [_one_synth(s) for s in specs]


class PlaceboSpaceResult(BaseModel):
    """Permutation-inference result for placebo-in-space (Abadie-Diamond-Hainmueller).

    Built on a :class:`SynthResult` whose already-validated ``synth()`` solver is
    re-run once per donor unit (each donor temporarily treated as the treated
    unit, with the remaining donors as its donor pool).

    All public outputs are named pandas objects, never raw arrays:

    * ``ratios`` -- ``pd.Series`` of ``post_mspe / pre_mspe`` per placebo *donor*
      unit (indexed by unit id), after any exclusion filter.
    * ``gap_paths`` -- ``pd.DataFrame`` of each placebo run's gap path, indexed
      by time period, one column per placebo donor (needed by a future plotting
      pass; plotting itself stays out of scope here).
    * ``p_value`` -- the ADH permutation p-value (fraction of placebo ratios
      ``>=`` the treated unit's ratio).

    Immutability, ``.tidy()`` / ``.summary()`` / ``.export()`` follow the
    project's standard result contract.  ``.plot()`` and ``.predict()`` remain
    ``NotImplementedError`` (deliberately out of scope).
    """

    def __init__(
        self,
        *,
        treated_unit: Any,
        treated_ratio: float,
        treated_pre_mspe: float,
        treated_post_mspe: float,
        ratios: pd.Series,
        excluded: list[dict[str, Any]],
        pre_mspe_threshold: Optional[float],
        p_value: float,
        gap_paths: pd.DataFrame,
        pre_mspe: pd.Series,
        post_mspe: pd.Series,
        call: dict[str, Any],
    ) -> None:
        self.formula = f"placebo_space({treated_unit!r})"
        self.data_shape = (len(ratios) + 1, 1)
        self.cov_type = "synth placebo (space)"
        self.call = call
        self.timestamp = __import__("datetime").datetime.now()
        self.package_version = __version__

        self.treated_unit = treated_unit
        self.treated_ratio = float(treated_ratio)
        self.treated_pre_mspe = float(treated_pre_mspe)
        self.treated_post_mspe = float(treated_post_mspe)
        self.ratios = ratios
        self.excluded = list(excluded)
        self.pre_mspe_threshold = pre_mspe_threshold
        self.p_value = float(p_value)
        self.gap_paths = gap_paths
        self.pre_mspe = pre_mspe
        self.post_mspe = post_mspe

        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Per-unit MSPE / ratio table (treated unit included as the first row)."""
        df = pd.DataFrame(
            {
                "unit": [self.treated_unit, *self.ratios.index.tolist()],
                "is_treated": [True, *[False] * len(self.ratios)],
                "pre_mspe": [self.treated_pre_mspe, *self.pre_mspe.tolist()],
                "post_mspe": [self.treated_post_mspe, *self.post_mspe.tolist()],
                "mspe_ratio": [self.treated_ratio, *self.ratios.tolist()],
            }
        )
        df.index.name = None
        return df

    def summary(self) -> str:
        header = (
            f"        Synthetic Control Placebo-in-Space (ADH) Results        \n"
            f"================================================================\n"
            f"Treated unit:               {self.treated_unit}\n"
            f"Placebo units (retained):   {len(self.ratios)}\n"
            f"Placebo units (excluded):   {len(self.excluded)}\n"
            f"Pre-MSPE exclusion mult:    "
            f"{self.pre_mspe_threshold if self.pre_mspe_threshold is not None else 'none'}\n"
            f"Treated unit ratio:         {self.treated_ratio:.6e}\n"
            f"Permutation p-value:        {self.p_value:.6f}\n"
            f"================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n================================================================\n"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["kind"] = "space"
        d["treated_unit"] = self.treated_unit
        d["treated_ratio"] = self.treated_ratio
        d["treated_pre_mspe"] = self.treated_pre_mspe
        d["treated_post_mspe"] = self.treated_post_mspe
        d["p_value"] = self.p_value
        d["n_placebos"] = len(self.ratios)
        d["n_excluded"] = len(self.excluded)
        d["pre_mspe_threshold"] = self.pre_mspe_threshold
        d["ratios"] = {str(k): float(v) for k, v in self.ratios.items()}
        d["excluded"] = [
            {
                "unit": e.get("unit"),
                "pre_mspe": float(e["pre_mspe"]) if e.get("pre_mspe") == e.get("pre_mspe") else None,
                "ratio": float(e["ratio"]) if e.get("ratio") == e.get("ratio") else None,
                "reason": e.get("reason"),
            }
            for e in self.excluded
        ]
        return d


class PlaceboTimeResult(BaseModel):
    """Permutation-inference result for placebo-in-time (Abadie-Diamond-Hainmueller).

    Built on a :class:`SynthResult` whose already-validated ``synth()`` solver is
    re-run once per candidate *pre-treatment* date: each candidate date is
    treated as a pseudo-treatment date, with the post window starting at the
    period immediately after it and the donor pool unchanged.

    All public outputs are named pandas objects, never raw arrays:

    * ``ratios`` -- ``pd.Series`` of ``post_mspe / pre_mspe`` per candidate
      pre-treatment date (indexed by that date).
    * ``gap_paths`` -- ``pd.DataFrame`` of each placebo run's gap path, indexed
      by time period, one column per candidate date (needed by a future plotting
      pass; plotting itself stays out of scope here).
    * ``p_value`` -- the ADH permutation p-value (fraction of candidate-date
      ratios ``>=`` the treated unit's actual ratio).

    Immutability, ``.tidy()`` / ``.summary()`` / ``.export()`` follow the
    project's standard result contract.  ``.plot()`` and ``.predict()`` remain
    ``NotImplementedError`` (deliberately out of scope).
    """

    def __init__(
        self,
        *,
        treated_unit: Any,
        treated_ratio: float,
        treated_pre_mspe: float,
        treated_post_mspe: float,
        ratios: pd.Series,
        excluded: list[dict[str, Any]],
        p_value: float,
        gap_paths: pd.DataFrame,
        pre_mspe: pd.Series,
        post_mspe: pd.Series,
        call: dict[str, Any],
    ) -> None:
        self.formula = f"placebo_time({treated_unit!r})"
        self.data_shape = (len(ratios) + 1, 1)
        self.cov_type = "synth placebo (time)"
        self.call = call
        self.timestamp = __import__("datetime").datetime.now()
        self.package_version = __version__

        self.treated_unit = treated_unit
        self.treated_ratio = float(treated_ratio)
        self.treated_pre_mspe = float(treated_pre_mspe)
        self.treated_post_mspe = float(treated_post_mspe)
        self.ratios = ratios
        self.excluded = list(excluded)
        self.p_value = float(p_value)
        self.gap_paths = gap_paths
        self.pre_mspe = pre_mspe
        self.post_mspe = post_mspe

        self._freeze()

    def tidy(self) -> pd.DataFrame:
        """Per-candidate-date MSPE / ratio table."""
        df = pd.DataFrame(
            {
                "period": self.ratios.index.tolist(),
                "pre_mspe": self.pre_mspe.tolist(),
                "post_mspe": self.post_mspe.tolist(),
                "mspe_ratio": self.ratios.tolist(),
            }
        )
        df.index.name = None
        return df

    def summary(self) -> str:
        header = (
            f"        Synthetic Control Placebo-in-Time (ADH) Results         \n"
            f"================================================================\n"
            f"Treated unit:               {self.treated_unit}\n"
            f"Candidate dates (used):     {len(self.ratios)}\n"
            f"Candidate dates (excluded): {len(self.excluded)}\n"
            f"Treated unit ratio:         {self.treated_ratio:.6e}\n"
            f"Permutation p-value:        {self.p_value:.6f}\n"
            f"================================================================\n"
        )
        tbl = self.tidy().to_string(index=False)
        return header + tbl + "\n================================================================\n"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["kind"] = "time"
        d["treated_unit"] = self.treated_unit
        d["treated_ratio"] = self.treated_ratio
        d["treated_pre_mspe"] = self.treated_pre_mspe
        d["treated_post_mspe"] = self.treated_post_mspe
        d["p_value"] = self.p_value
        d["n_placebos"] = len(self.ratios)
        d["n_excluded"] = len(self.excluded)
        d["ratios"] = {str(k): float(v) for k, v in self.ratios.items()}
        d["excluded"] = [
            {
                "time": e.get("time"),
                "pre_mspe": float(e["pre_mspe"]) if e.get("pre_mspe") == e.get("pre_mspe") else None,
                "ratio": float(e["ratio"]) if e.get("ratio") == e.get("ratio") else None,
                "reason": e.get("reason"),
            }
            for e in self.excluded
        ]
        return d


def placebo_space(
    result: SynthResult,
    data: pd.DataFrame,
    *,
    exclude_pre_mspe_multiple: Optional[float] = None,
    parallel: bool = False,
    **solver_kwargs: Any,
) -> PlaceboSpaceResult:
    """Placebo-in-space permutation inference for a fitted synthetic control.

    Re-runs the already-validated :func:`synth` solver once per donor unit,
    treating each donor temporarily as the treated unit (with the remaining
    donors as its donor pool).  This mirrors the manual ADH placebo-in-space
    loop and reuses the validated core engine rather than duplicating it.

    Parameters
    ----------
    result : SynthResult
        The original fit.  Its stored configuration (``outcome``,
        ``treated_unit``, ``donor_pool``, ``entity``, ``time``, ``pre_period``,
        ``post_period``, ``predictors``) is used to reconstruct each placebo
        call, so the caller only supplies the panel ``data``.
    data : pd.DataFrame
        The full balanced panel (the same one used for the original fit).
    exclude_pre_mspe_multiple : float, optional
        If given, placebo units whose *pre*-treatment MSPE exceeds this multiple
        of the treated unit's pre-treatment MSPE are dropped from the ratio set
        and the p-value (and from ``gap_paths``).  ADH discuss excluding
        poorly-pre-fitting placebos for exactly this reason; the default
        ``None`` performs **no** exclusion (transparent behavior).  ADH
        applications typically pass a multiple such as 10.
    parallel : bool, default False
        Run the per-donor ``synth()`` fits across a process pool.  Each fit is an
        independent pure call, so results are **bit-identical** to the sequential
        path; this only changes wall-clock time.  Useful for large donor pools
        (the loop is sequential below ``_MIN_PARALLEL_ITEMS`` to avoid spawn
        overhead on small panels).  Note: a thread pool gives no speedup here
        (the SLSQP/QM path holds the GIL), so this uses processes.
    **solver_kwargs
        Forwarded to :func:`synth`'s inner/outer ``scipy.optimize.minimize``.

    Returns
    -------
    PlaceboSpaceResult
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    outcome = result.outcome
    treated = result.treated_unit
    donors = list(result.donor_pool)
    entity = result.entity
    time = result.time
    pre_period = result.pre_period
    post_period = result.post_period
    predictors = result.predictors  # None or list[str]

    treated_pre = result.pre_mspe
    treated_post = result.post_mspe
    treated_ratio = (treated_post / treated_pre) if treated_pre > 0 else float("inf")

    ratio_map: dict[Any, float] = {}
    pre_map: dict[Any, float] = {}
    post_map: dict[Any, float] = {}
    gap_map: dict[Any, pd.Series] = {}
    excluded: list[dict[str, Any]] = []

    specs = [
        {
            "__key__": d,
            "data": data,
            "outcome": outcome,
            "treated_unit": d,
            "donor_pool": [u for u in donors if u != d],
            "entity": entity,
            "time": time,
            "pre_period": pre_period,
            "post_period": post_period,
            "predictors": predictors,
            **solver_kwargs,
        }
        for d in donors
    ]
    for res in _parmap(specs, parallel):
        d = res["key"]
        if not res["ok"]:
            # a donor that cannot be fit as a placebo is recorded, never
            # silently dropped.
            excluded.append(
                {"unit": d, "pre_mspe": float("nan"),
                 "ratio": float("nan"), "reason": f"fit failed: {res['error']}"}
            )
            continue
        pre_mspe = res["pre_mspe"]
        post_mspe = res["post_mspe"]
        ratio = (post_mspe / pre_mspe) if pre_mspe > 0 else float("inf")
        if (
            exclude_pre_mspe_multiple is not None
            and pre_mspe
            > exclude_pre_mspe_multiple * (treated_pre if treated_pre > 0 else 0.0)
        ):
            excluded.append({"unit": d, "pre_mspe": pre_mspe, "ratio": ratio})
            continue
        ratio_map[d] = ratio
        pre_map[d] = pre_mspe
        post_map[d] = post_mspe
        gap_map[d] = res["gap"]

    ratios = pd.Series(ratio_map, name="mspe_ratio")
    if ratios.index.name is None:
        ratios.index.name = entity
    pre_mspe_s = pd.Series(pre_map, name="pre_mspe")
    if pre_mspe_s.index.name is None:
        pre_mspe_s.index.name = entity
    post_mspe_s = pd.Series(post_map, name="post_mspe")
    if post_mspe_s.index.name is None:
        post_mspe_s.index.name = entity
    gap_paths = pd.DataFrame(gap_map)
    if not gap_paths.empty:
        gap_paths.index.name = time

    p_value = _adh_permutation_p_value(ratios, treated_ratio)

    call = _capture_call(
        kind="space", treated_unit=treated, n_donors=len(donors),
        exclude_pre_mspe_multiple=exclude_pre_mspe_multiple,
        model_type="synth_placebo", **solver_kwargs,
    )

    return PlaceboSpaceResult(
        treated_unit=treated,
        treated_ratio=treated_ratio,
        treated_pre_mspe=treated_pre,
        treated_post_mspe=treated_post,
        ratios=ratios,
        excluded=excluded,
        pre_mspe_threshold=exclude_pre_mspe_multiple,
        p_value=p_value,
        gap_paths=gap_paths,
        pre_mspe=pre_mspe_s,
        post_mspe=post_mspe_s,
        call=call,
    )


def placebo_time(
    result: SynthResult,
    data: pd.DataFrame,
    *,
    parallel: bool = False,
    **solver_kwargs: Any,
) -> PlaceboTimeResult:
    """Placebo-in-time permutation inference for a fitted synthetic control.

    Re-runs the already-validated :func:`synth` solver once per candidate
    *pre-treatment* date: each candidate date ``t_c`` is treated as a
    pseudo-treatment date, with the post window starting at the period
    immediately after ``t_c`` and the donor pool unchanged.  This mirrors the
    manual ADH placebo-in-time test and reuses the validated core engine.

    Parameters
    ----------
    result : SynthResult
        The original fit.  Its stored configuration is used to reconstruct each
        placebo call, so the caller only supplies the panel ``data``.
    data : pd.DataFrame
        The full balanced panel (the same one used for the original fit).
    parallel : bool, default False
        Run the per-candidate-date ``synth()`` fits across a process pool.  Each
        fit is an independent pure call, so results are **bit-identical** to the
        sequential path; this only changes wall-clock time.  See
        :func:`placebo_space` for the parallelization rationale and the small-
        panel threshold.
    **solver_kwargs
        Forwarded to :func:`synth`'s inner/outer ``scipy.optimize.minimize``.

    Returns
    -------
    PlaceboTimeResult

    Notes
    -----
    ``exclude_pre_mspe_multiple`` is intentionally **not** accepted here: in the
    in-time loop, "pre-MSPE" is the treated unit's own fit against itself at a
    different candidate cutoff, not a fit-quality comparison across independent
    placebo units, so the space-style exclusion concept does not apply.  Passing
    it raises ``TypeError`` (rejected explicitly before any ``synth`` call, so
    the space-only parameter is never silently reinterpreted as a solver kwarg).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # ``exclude_pre_mspe_multiple`` is space-only (see the module docstring and
    # the Notes above).  ``**solver_kwargs`` would otherwise swallow it and
    # forward it silently to ``synth`` (which ignores unknown kwargs), so we
    # reject it loudly here rather than accepting and reinterpreting it.
    if "exclude_pre_mspe_multiple" in solver_kwargs:
        raise TypeError(
            "exclude_pre_mspe_multiple is space-only (use placebo_space); "
            "placebo_time does not accept it because, in the in-time loop, "
            "'pre-MSPE' is the treated unit's own fit against itself at a "
            "different candidate cutoff, not a fit-quality comparison across "
            "independent placebo units."
        )

    outcome = result.outcome
    treated = result.treated_unit
    donors = list(result.donor_pool)
    entity = result.entity
    time = result.time
    pre_period = result.pre_period
    predictors = result.predictors  # None or list[str]

    treated_pre = result.pre_mspe
    treated_post = result.post_mspe
    treated_ratio = (treated_post / treated_pre) if treated_pre > 0 else float("inf")

    all_times = sorted(pd.unique(data[time]))
    if pre_period not in all_times:
        raise ValueError(
            f"pre_period={pre_period!r} is not present in the '{time}' column."
        )
    i_pre = all_times.index(pre_period)
    # Candidate pre-treatment dates: every period strictly before pre_period.
    candidates = all_times[:i_pre]
    # Alignment index for the gap paths: all periods up to the actual pre_period.
    align_index = all_times[: i_pre + 1]

    ratio_map: dict[Any, float] = {}
    pre_map: dict[Any, float] = {}
    post_map: dict[Any, float] = {}
    gap_map: dict[Any, pd.Series] = {}
    excluded: list[dict[str, Any]] = []

    specs = [
        {
            "__key__": t_c,
            "data": data,
            "outcome": outcome,
            "treated_unit": treated,
            "donor_pool": donors,
            "entity": entity,
            "time": time,
            "pre_period": t_c,
            "post_period": all_times[all_times.index(t_c) + 1],
            "predictors": predictors,
            **solver_kwargs,
        }
        for t_c in candidates
    ]
    for res in _parmap(specs, parallel):
        t_c = res["key"]
        if not res["ok"]:
            # a candidate date that cannot be fit is recorded, never silently
            # dropped.
            excluded.append(
                {"time": t_c, "pre_mspe": float("nan"),
                 "ratio": float("nan"), "reason": f"fit failed: {res['error']}"}
            )
            continue
        pre_mspe = res["pre_mspe"]
        post_mspe = res["post_mspe"]
        ratio = (post_mspe / pre_mspe) if pre_mspe > 0 else float("inf")
        # Reindex the pseudo-fit's gap path onto the alignment index
        # (periods <= the actual pre_period), filling nothing (each candidate
        # window already covers this range with no gaps).
        gap_map[t_c] = res["gap"].reindex(align_index)
        ratio_map[t_c] = ratio
        pre_map[t_c] = pre_mspe
        post_map[t_c] = post_mspe

    ratios = pd.Series(ratio_map, name="mspe_ratio")
    if ratios.index.name is None:
        ratios.index.name = time
    pre_mspe_s = pd.Series(pre_map, name="pre_mspe")
    if pre_mspe_s.index.name is None:
        pre_mspe_s.index.name = time
    post_mspe_s = pd.Series(post_map, name="post_mspe")
    if post_mspe_s.index.name is None:
        post_mspe_s.index.name = time
    gap_paths = pd.DataFrame(gap_map)
    if not gap_paths.empty:
        gap_paths.index.name = time

    p_value = _adh_permutation_p_value(ratios, treated_ratio)

    call = _capture_call(
        kind="time", treated_unit=treated, n_candidates=len(candidates),
        model_type="synth_placebo", **solver_kwargs,
    )

    return PlaceboTimeResult(
        treated_unit=treated,
        treated_ratio=treated_ratio,
        treated_pre_mspe=treated_pre,
        treated_post_mspe=treated_post,
        ratios=ratios,
        excluded=excluded,
        p_value=p_value,
        gap_paths=gap_paths,
        pre_mspe=pre_mspe_s,
        post_mspe=post_mspe_s,
        call=call,
    )
