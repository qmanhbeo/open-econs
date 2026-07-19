"""Thin backend wrapping ``MASS::rlm`` (R) for robust M-/MM-estimation.

rpy2 is not a hard dependency of open-econs.  When R is available on the
machine (``Rscript`` on ``PATH`` or ``R_EXE``), we call ``MASS::rlm`` exactly
once per fit via a small subprocess and parse its JSON output.

This backend underpins ``oe.robust_reg(..., parity="rlm")`` — the validated
R parity branch.  Coefficients, the ``cov.unscaled * s^2`` covariance, scale,
weights, and residuals match ``MASS::rlm(method="MM"/"M", psi=psi.bisquare,
init="ls", scale.est="MAD")`` to 1e-6.  The ``parity="stata"`` branch in
``open_econs.models.linear.robust_reg`` is a pure-Python bisquare M-estimator
and does NOT use this backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# Default Rscript path. Override with R_EXE env var.
R_EXE = os.environ.get(
    "R_EXE",
    r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe",
)


def r_available() -> bool:
    """Return True if the Rscript binary exists on this machine."""
    return Path(R_EXE).is_file()


# R template.  {formula}, {method}, {maxit}, {acc} are filled in by the caller.
# Outputs JSON: b (coef), se_vcov (sqrt diag of cov.unscaled*s^2), V (k x k),
# scale, w (weights), resid, rss, nobs.
_R_TEMPLATE = r'''
library(MASS); library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
in_csv  <- args[1]
out_json <- args[2]
df <- read.csv(in_csv)
fit <- rlm({formula}, data = df, method = "{method}", psi = psi.bisquare,
           init = "ls", scale.est = "MAD", maxit = {maxit}, acc = {acc})
b <- as.numeric(coef(fit))
V <- summary(fit)$cov.unscaled * fit$s^2
V <- as.matrix(V)
# Re-order V to match coefficient order (it already does via coef names).
se <- sqrt(diag(V))
scale <- as.numeric(fit$s)
w <- as.numeric(fit$w)
resid <- as.numeric(fit$resid)
rss <- as.numeric(sum(resid^2))
nobs <- as.numeric(length(resid))
out <- list(
  b = b,
  V = V,
  se = se,
  scale = scale,
  w = w,
  resid = resid,
  rss = rss,
  nobs = nobs,
  names = names(coef(fit))
)
write_json(out, out_json, digits = 15, auto_unbox = FALSE, pretty = TRUE)
'''


def rlm_fit(
    formula: str,
    csv_path: str,
    method: str = "MM",
    maxit: int = 20,
    acc: float = 1e-4,
) -> dict[str, Any]:
    """Fit ``MASS::rlm`` via Rscript and return the parsed result dict.

    Parameters
    ----------
    formula : str
        R-style two-sided formula, e.g. ``"y ~ x1 + x2"``.
    csv_path : str
        Path to a CSV containing every variable referenced by *formula*.
    method : {"MM", "M"}
        ``"MM"`` → ``rlm(method="MM")`` (bisquare S-estimate init); ``"M"`` →
        ``rlm(method="M")`` (plain bisquare M-estimator, MAD scale).
    maxit, acc : int, float
        Iteration controls passed to ``rlm``.

    Returns
    -------
    dict
        Keys: ``b``, ``V``, ``se``, ``scale``, ``w``, ``resid``, ``rss``,
        ``nobs``, ``names``.
    """
    if not r_available():
        raise RuntimeError(
            "R (Rscript) is not available; cannot run the MASS::rlm backend. "
            "Install R or use the pure-Python fallback path."
        )

    r_script = _R_TEMPLATE.format(
        formula=formula, method=method, maxit=int(maxit), acc=repr(float(acc)),
    )
    with tempfile.TemporaryDirectory() as tmp:
        r_path = Path(tmp) / "rlm_fit.R"
        out_path = Path(tmp) / "rlm_fit.json"
        r_path.write_text(r_script, encoding="utf-8")
        result = subprocess.run(
            [R_EXE, str(r_path), csv_path, str(out_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(csv_path).resolve().parent),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Rscript (MASS::rlm) exited with code {result.returncode}\n"
                f"stderr: {result.stderr[:3000]}"
            )
        with out_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
