"""Dual-mode Stata runner for parity testing.

Three modes, controlled by whether StataMP is installed and by the
``OE_REGENERATE_FIXTURES`` environment variable:

  1. **Read committed fixture (default):** StataMP not launched. The committed
     ``.dta`` is read directly as ground truth. This is what happens on a normal
     ``pytest`` run (the test is also deselected by the ``stata`` marker unless
     ``-m "stata or r"`` is passed) and on CI.
  2. **Regenerate fixture:** ``OE_REGENERATE_FIXTURES`` set truthy *and* StataMP
     installed → run the ``.do`` file and overwrite the ``.dta``. Use only when a
     ``.do`` script changed or right before a release.
  3. **Fallback (no Stata):** StataMP absent → no-op, committed ``.dta`` used.

In every mode a drift check verifies that the .do file has not been edited
since the .dta was last regenerated.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pandas as pd

# Default StataMP path. Override with STATA_EXE env var.
STATA_EXE = os.environ.get(
    "STATA_EXE",
    r"C:\Program Files\Stata17\StataMP-64.exe",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DO_DIR = Path(__file__).resolve().parent / "generate-fixtures"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class StataError(RuntimeError):
    """Raised when StataMP exits with a non-zero code."""


def stata_available() -> bool:
    """Return True if StataMP exists on this machine."""
    return Path(STATA_EXE).is_file()


def _check_drift(label: str) -> None:
    """Fail loudly if a .do file is newer than its .dta — the fixture is stale."""
    do_path = DO_DIR / f"{label}.do"
    dta_path = DO_DIR / f"{label}.dta"
    if not do_path.exists() or not dta_path.exists():
        return
    do_mtime = do_path.stat().st_mtime
    dta_mtime = dta_path.stat().st_mtime
    if do_mtime > dta_mtime:
        raise RuntimeError(
            f"STALE FIXTURE: {label}.do (mtime {time.ctime(do_mtime)}) is newer "
            f"than {label}.dta (mtime {time.ctime(dta_mtime)}).  "
            f"Regenerate with Stata or revert the .do change."
        )


def run_do(label: str) -> None:
    """Run tests/stata/generate-fixtures/{label}.do via StataMP to regenerate its .dta fixture.

    Regeneration is gated behind the ``OE_REGENERATE_FIXTURES`` environment
    variable: StataMP is only launched when that variable is set to a truthy
    value.  This keeps routine local test runs (and CI) free of any Stata
    binary invocation — they read the committed .dta fixtures as ground truth
    instead.  Set ``OE_REGENERATE_FIXTURES=1`` only when a .do script changed
    or right before a release.

    If StataMP is not available (or the gate is off), this is a no-op — the
    committed .dta is assumed to be current (drift check will catch stale
    fixtures).
    """
    if not stata_available():
        return
    if not os.environ.get("OE_REGENERATE_FIXTURES"):
        return

    do_path = DO_DIR / f"{label}.do"
    if not do_path.exists():
        raise FileNotFoundError(f"DO file not found: {do_path}")

    result = subprocess.run(
        [STATA_EXE, "-e", "do", str(do_path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise StataError(
            f"StataMP exited with code {result.returncode}\n"
            f"stderr: {result.stderr[:2000]}"
        )


def read_stata(label: str) -> dict[str, float]:
    """Run the .do file (if Stata is available), check drift, and read .dta.

    Returns a dict mapping ``name`` → ``value`` from the Stata output.
    """
    run_do(label)
    _check_drift(label)
    dta_path = DO_DIR / f"{label}.dta"
    if not dta_path.exists():
        raise FileNotFoundError(
            f"No .dta fixture found: {dta_path}.  "
            f"Run Stata to generate it, or commit the .dta file."
        )
    df = pd.read_stata(dta_path)
    return dict(zip(df["name"], df["value"]))
