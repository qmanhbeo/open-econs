"""Dual-mode R runner for parity testing.

Exact structural analog of ``tests/stata/stata_runner.py``, but for R
(``Rscript``) backed parity tests instead of Stata.

Three modes, controlled by whether R is installed and by the
``OE_REGENERATE_FIXTURES`` environment variable:

   1. **Read committed fixture (default):** Rscript not launched. The committed
      ``.json`` is read directly as ground truth. This is what happens on a
      normal ``pytest`` run (the test is also deselected by the ``r`` marker
      unless ``-m "stata or r"`` is passed) and on CI.
   2. **Regenerate fixture:** ``OE_REGENERATE_FIXTURES`` set truthy *and* R
      installed → run the ``.R`` file and overwrite the ``.json``. Use only
      when an ``.R`` script changed or right before a release.
   3. **Fallback (no R):** R absent → no-op, committed ``.json`` used.

In every mode a drift check verifies that the ``.R`` file has not been edited
since the ``.json`` was last regenerated.

The ``.R`` scripts and their expected-output ``.json`` fixtures, plus the
committed input ``.csv`` files that *both* the script and the Python test read,
live under this package:

    tests/r/
      do/            <label>.R            (scripts; read argv[1]=input csv, argv[2]=output json)
      fixtures/      <label>_input.csv    (committed input, read by BOTH sides)
                    <label>.json          (committed expected output, ground truth)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

# Default Rscript path. Override with R_EXE env var.
# NOTE: 4.6.1 is the version actually installed on the dev machine. The stale
# 4.5.2 paths hardcoded in the legacy test files were a bug; this is the single
# source of truth for the R binary.
R_EXE = os.environ.get(
    "R_EXE",
    r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe",
)

THIS_DIR = Path(__file__).resolve().parent
R_SCRIPT_DIR = THIS_DIR / "do"
R_FIXTURES_DIR = THIS_DIR / "fixtures"


class RError(RuntimeError):
    """Raised when Rscript exits with a non-zero code."""


def r_available() -> bool:
    """Return True if the Rscript binary exists on this machine."""
    return Path(R_EXE).is_file()


def _check_drift_r(label: str) -> None:
    """Fail loudly if a .R file is newer than its .json -- the fixture is stale.

    Mirrors ``tests/stata/stata_runner._check_drift`` exactly (strict ``>``).
    """
    r_path = R_SCRIPT_DIR / f"{label}.R"
    json_path = R_FIXTURES_DIR / f"{label}.json"
    if not r_path.exists() or not json_path.exists():
        return
    r_mtime = r_path.stat().st_mtime
    json_mtime = json_path.stat().st_mtime
    if r_mtime > json_mtime:
        raise RuntimeError(
            f"STALE FIXTURE: {label}.R (mtime {time.ctime(r_mtime)}) is newer "
            f"than {label}.json (mtime {time.ctime(json_mtime)}).  "
            f"Regenerate with R or revert the .R change."
        )


def run_r(label: str) -> None:
    """Run tests/r/do/{label}.R via Rscript to regenerate its .json fixture.

    Regeneration is gated behind the ``OE_REGENERATE_FIXTURES`` environment
    variable: Rscript is only launched when that variable is set to a truthy
    value.  This keeps routine local test runs (and CI) free of any R binary
    invocation -- they read the committed ``.json`` fixtures as ground truth
    instead.  Set ``OE_REGENERATE_FIXTURES=1`` only when an ``.R`` script
    changed or right before a release.

    If R is not available (or the gate is off), this is a no-op -- the
    committed ``.json`` is assumed to be current (drift check will catch stale
    fixtures).

    The script receives two positional arguments:
        argv[1] = tests/r/fixtures/{label}_input.csv   (committed input, read by both sides)
        argv[2] = tests/r/fixtures/{label}.json        (expected output to (over)write)
    """
    if not r_available():
        return
    if not os.environ.get("OE_REGENERATE_FIXTURES"):
        return

    r_path = R_SCRIPT_DIR / f"{label}.R"
    if not r_path.exists():
        raise FileNotFoundError(f"R script not found: {r_path}")

    in_csv = R_FIXTURES_DIR / f"{label}_input.csv"
    out_json = R_FIXTURES_DIR / f"{label}.json"
    result = subprocess.run(
        [R_EXE, str(r_path), str(in_csv), str(out_json)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RError(
            f"Rscript exited with code {result.returncode}\n"
            f"stderr: {result.stderr[:2000]}"
        )


def read_r(label: str) -> dict:
    """Run the .R file (if R is available), check drift, and read .json.

    Returns the parsed JSON dict (ground truth from R).
    """
    run_r(label)
    _check_drift_r(label)
    json_path = R_FIXTURES_DIR / f"{label}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"No .json fixture found: {json_path}.  "
            f"Run R to generate it, or commit the .json file."
        )
    with json_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
