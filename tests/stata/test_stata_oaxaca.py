"""Stata parity tests for Oaxaca-Blinder (SSC: oaxaca v4.1.1, Ben Jann).

Each test class exercises one decomposition variant and compares every
component against Stata's ``e(b)`` matrix (column positions verified
from ``oaxaca.ado`` lines 1680-1693).

Terminology mapping (Stata ↔ OE):
  Two-fold:  gap ↔ total_gap, explained ↔ explained, unexplained ↔ unexplained
  Three-fold: gap ↔ total_gap, endowment ↔ explained, coefficients ↔ unexplained,
              interaction ↔ interaction
"""

from __future__ import annotations

import numpy as np
import pytest

import open_econs as oe

from .stata_runner import read_stata

pytestmark = pytest.mark.stata

# Module-level Stata caches.
S_TWO_FOLD = read_stata("oaxaca_two_fold")
S_THREE_FOLD = read_stata("oaxaca_three_fold")

# Relative tolerances per component type.
# Gap and explained/endowment are large (~10, ~2); 1e-6 rtol is feasible.
# Unexplained/coefficients are larger (~8); 1e-6 rtol also feasible.
# Interaction can be near-zero (<<1 in typical data); use an absolute floor.
_RTOL_GAP = 1e-6
_RTOL_EXPLAINED = 1e-6
_RTOL_UNEXPLAINED = 1e-6
_ATOL_INTERACTION = 1e-6  # interaction can be small → absolute tolerance


def _check_component(oe_val: float, stata_name: str, stata: dict,
                     rtol: float = 1e-6, atol: float = 0.0) -> None:
    """Assert that *oe_val* matches *stata[stata_name]* within tolerance."""
    ref = stata[stata_name]
    ok = np.isclose(oe_val, ref, rtol=rtol, atol=atol)
    if not ok:
        rel_err = abs(oe_val - ref) / max(abs(ref), 1e-15)
        abs_err = abs(oe_val - ref)
        pytest.fail(
            f"{stata_name}: OE={oe_val:.10f}  Stata={ref:.10f}  "
            f"rel_err={rel_err:.2e}  abs_err={abs_err:.2e}"
        )


# ── Two-fold variants ──────────────────────────────────────────────────

class TestOaxacaTwoFoldPooled:
    """Two-fold decomposition with pooled reference (including group dummy)."""
    @pytest.fixture(autouse=True)
    def _run(self, df_oaxaca):
        self.s = S_TWO_FOLD
        self.oe_r = oe.oaxaca("y ~ edu + age + female", data=df_oaxaca,
                               by="female", decomposition_type="two-fold",
                               reference="pooled")

    def test_gap(self):
        _check_component(self.oe_r.total_gap, "pooled_gap", self.s, rtol=_RTOL_GAP)

    def test_explained(self):
        _check_component(self.oe_r.explained, "pooled_explained", self.s,
                         rtol=_RTOL_EXPLAINED)

    def test_unexplained(self):
        _check_component(self.oe_r.unexplained, "pooled_unexplained", self.s,
                         rtol=_RTOL_UNEXPLAINED)


class TestOaxacaTwoFoldOmega:
    """Two-fold decomposition with omega reference (Neumark, no group dummy)."""
    @pytest.fixture(autouse=True)
    def _run(self, df_oaxaca):
        self.s = S_TWO_FOLD
        self.oe_r = oe.oaxaca("y ~ edu + age + female", data=df_oaxaca,
                               by="female", decomposition_type="two-fold",
                               reference="omega")

    def test_gap(self):
        _check_component(self.oe_r.total_gap, "omega_gap", self.s, rtol=_RTOL_GAP)

    def test_explained(self):
        _check_component(self.oe_r.explained, "omega_explained", self.s,
                         rtol=_RTOL_EXPLAINED)

    def test_unexplained(self):
        _check_component(self.oe_r.unexplained, "omega_unexplained", self.s,
                         rtol=_RTOL_UNEXPLAINED)


class TestOaxacaTwoFoldRefGroup1:
    """Two-fold decomposition using group 1 coefficients as the non‑discriminatory reference (Stata ``weight(1)``)."""
    @pytest.fixture(autouse=True)
    def _run(self, df_oaxaca):
        self.s = S_TWO_FOLD
        self.oe_r = oe.oaxaca("y ~ edu + age + female", data=df_oaxaca,
                               by="female", decomposition_type="two-fold",
                               reference="group1")

    def test_gap(self):
        _check_component(self.oe_r.total_gap, "weight1_gap", self.s, rtol=_RTOL_GAP)

    def test_explained(self):
        _check_component(self.oe_r.explained, "weight1_explained", self.s,
                         rtol=_RTOL_EXPLAINED)

    def test_unexplained(self):
        _check_component(self.oe_r.unexplained, "weight1_unexplained", self.s,
                         rtol=_RTOL_UNEXPLAINED)


class TestOaxacaTwoFoldRefGroup2:
    """Two-fold decomposition using group 2 coefficients as the non‑discriminatory reference (Stata ``weight(0)``)."""
    @pytest.fixture(autouse=True)
    def _run(self, df_oaxaca):
        self.s = S_TWO_FOLD
        self.oe_r = oe.oaxaca("y ~ edu + age + female", data=df_oaxaca,
                               by="female", decomposition_type="two-fold",
                               reference="group2")

    def test_gap(self):
        _check_component(self.oe_r.total_gap, "weight0_gap", self.s, rtol=_RTOL_GAP)

    def test_explained(self):
        _check_component(self.oe_r.explained, "weight0_explained", self.s,
                         rtol=_RTOL_EXPLAINED)

    def test_unexplained(self):
        _check_component(self.oe_r.unexplained, "weight0_unexplained", self.s,
                         rtol=_RTOL_UNEXPLAINED)


# ── Three-fold variants ────────────────────────────────────────────────

class TestOaxacaThreeFoldDefault:
    """Three-fold decomposition (default; group 2 coefficients as reference)."""
    @pytest.fixture(autouse=True)
    def _run(self, df_oaxaca):
        self.s = S_THREE_FOLD
        self.oe_r = oe.oaxaca("y ~ edu + age + female", data=df_oaxaca,
                               by="female", decomposition_type="three-fold")

    def test_gap(self):
        _check_component(self.oe_r.total_gap, "threefold_gap", self.s,
                         rtol=_RTOL_GAP)

    def test_endowment(self):
        """Three-fold 'endowment' ↔ OE's ``.explained``."""
        _check_component(self.oe_r.explained, "threefold_endowment", self.s,
                         rtol=_RTOL_EXPLAINED)

    def test_coefficients(self):
        """Three-fold 'coefficients' ↔ OE's ``.unexplained``."""
        _check_component(self.oe_r.unexplained, "threefold_coefficients", self.s,
                         rtol=_RTOL_UNEXPLAINED)

    def test_interaction(self):
        _check_component(self.oe_r.interaction, "threefold_interaction", self.s,
                         rtol=_RTOL_EXPLAINED, atol=_ATOL_INTERACTION)


class TestOaxacaThreeFoldReverse:
    """Three-fold decomposition with group 1 coefficients as reference (Stata ``threefold(reverse)``)."""
    @pytest.fixture(autouse=True)
    def _run(self, df_oaxaca):
        self.s = S_THREE_FOLD
        self.oe_r = oe.oaxaca("y ~ edu + age + female", data=df_oaxaca,
                               by="female", decomposition_type="three-fold",
                               reverse=True)

    def test_gap(self):
        _check_component(self.oe_r.total_gap, "threefold_rev_gap", self.s,
                         rtol=_RTOL_GAP)

    def test_endowment(self):
        _check_component(self.oe_r.explained, "threefold_rev_endowment", self.s,
                         rtol=_RTOL_EXPLAINED)

    def test_coefficients(self):
        _check_component(self.oe_r.unexplained, "threefold_rev_coefficients", self.s,
                         rtol=_RTOL_UNEXPLAINED)

    def test_interaction(self):
        _check_component(self.oe_r.interaction, "threefold_rev_interaction", self.s,
                         rtol=_RTOL_EXPLAINED, atol=_ATOL_INTERACTION)
