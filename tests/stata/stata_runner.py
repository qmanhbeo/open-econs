"""Thin wrapper around StataMP for parity testing.

The workflow:
  1. Each estimator has a hand-written .do file in tests/stata/do/
  2. Python calls StataMP to run the .do file
  3. The .do file saves results as a .dta file
  4. Python reads the .dta file with pd.read_stata()
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Default StataMP path. Override with STATA_EXE env var.
STATA_EXE = os.environ.get(
    "STATA_EXE",
    r"C:\Program Files\Stata17\StataMP-64.exe",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DO_DIR = Path(__file__).resolve().parent / "do"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class StataError(RuntimeError):
    """Raised when StataMP exits with a non-zero code."""


def stata_available() -> bool:
    """Return True if StataMP exists on this machine."""
    return Path(STATA_EXE).is_file()


def run_do(label: str) -> None:
    """Run tests/stata/do/{label}.do via StataMP.

    The .do file is expected to save its output as
    tests/stata/do/{label}.dta (or {label}_results.dta).
    """
    if not stata_available():
        raise StataError(f"StataMP not found at {STATA_EXE}")

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
