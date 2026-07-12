"""Shared validation/normalization for the ``cov_type`` parameter.

Every estimator that accepts a ``cov_type`` string should route it through
:func:`validate_cov_type` so that:

* an invalid value raises a single, consistent, open-econs-native ``ValueError``
  (instead of an opaque statsmodels/linearmodels internal error, or a silent
  fallback to a default that masks the typo);
* the lowercase spelling ``"hac"`` is accepted as an alias for the project's
  preferred uppercase ``"HAC"`` -- but *only* where ``"HAC"`` is already a
  valid option for that estimator.

Casing decision (narrow, deliberate)
------------------------------------
The HAC alias is the *only* case-insensitivity this helper introduces.  Only
``"hac"`` (in any mixed-case variant such as ``"Hac"``) is normalized to
``"HAC"``.  No other ``cov_type`` string is case-normalized -- so ``"hc2"`` is
still rejected even though ``"HC2"`` is valid, and ``"nonrobust"`` must be
spelled exactly.  This is intentional: the ``"hac"``/``"HAC"`` mismatch was one
of the two explicitly flagged cross-estimator inconsistencies, whereas
blanket case-insensitivity for every ``cov_type`` would quietly change
behavior for values nobody flagged.  Any *other* spelling normalization (e.g.
``did``/``event_study`` historically accept ``"robust"`` as an alias for
``"HC2"``) must be passed explicitly via the ``aliases`` map, so the precedent
is visible at the call site rather than hidden in the helper.
"""

from __future__ import annotations

from typing import Iterable


def validate_cov_type(
    cov_type: str,
    *,
    accepted: Iterable[str],
    estimator: str,
    aliases: dict[str, str] | None = None,
) -> str:
    """Validate and normalize a ``cov_type`` string.

    Parameters
    ----------
    cov_type : str
        The user-supplied covariance-type string.
    accepted : Iterable[str]
        The set of canonical ``cov_type`` values the estimator actually
        supports.  Values not in this set raise ``ValueError``.
    estimator : str
        Human-readable estimator name used in the error message (e.g.
        ``"ols()"`` or ``"PanelContext.driscoll_kraay()"``).  Names the
        estimator so a user can tell which call failed.
    aliases : dict[str, str], optional
        Estimator-specific aliases that map a user spelling to a canonical
        value already present in ``accepted`` (e.g. ``{"robust": "HC2"}`` used
        by ``did``/``event_study``).  These are applied *after* the HAC casing
        rule and are the only non-HAC normalizations permitted.

    Returns
    -------
    str
        The canonical (possibly normalized) ``cov_type`` to use downstream.

    Raises
    ------
    ValueError
        If the (normalized) value is not in ``accepted``.
    """
    if not isinstance(cov_type, str):
        raise ValueError(
            f"{estimator}: cov_type must be a string, got {cov_type!r}."
        )

    # Narrow, deliberate exception: only "hac" (any case) -> "HAC".
    if cov_type.lower() == "hac":
        norm = "HAC"
    elif aliases and cov_type in aliases:
        norm = aliases[cov_type]
    else:
        norm = cov_type

    accepted_set = set(accepted)
    if norm not in accepted_set:
        allowed = ", ".join(repr(a) for a in sorted(accepted_set))
        raise ValueError(
            f"{estimator}: invalid cov_type {cov_type!r}. "
            f"Accepted values are: {allowed}."
        )
    return norm
