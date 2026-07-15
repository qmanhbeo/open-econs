"""R parity tests for VAR lag-order selection (``var_select_order``).

IC convention
-------------
OE's default ``var_select_order()`` uses the standard IC convention (all
parameters in the penalty), matching R ``VARselect`` and statsmodels.

Fixture
-------
``tests/r/fixtures/expected/var_basic.json`` stores ``selected_lag``
from R ``VARselect(y, lag.max=5, type="const")``.
"""

from __future__ import annotations

import pandas as pd
import pytest

import open_econs as oe

from ..r_runner import read_r

pytestmark = pytest.mark.r

# Module-level fixture cache
R_VAR = read_r("var_basic")
DF_VAR_INPUT = pd.read_csv("tests/r/fixtures/inputs/var_input.csv")


class TestVARLagOrderRIC:
    """OE standard IC selected lags vs R ``VARselect``."""

    @pytest.fixture(scope="class")
    def result(self):
        return oe.var_select_order(DF_VAR_INPUT, maxlags=5, trend="c")

    def test_aic_selected_lag(self, result):
        assert result.selected["aic"] == R_VAR["selected_lag"]["aic"]

    def test_bic_selected_lag(self, result):
        assert result.selected["bic"] == R_VAR["selected_lag"]["bic"]

    def test_hqic_selected_lag(self, result):
        assert result.selected["hqic"] == R_VAR["selected_lag"]["hqic"]

    def test_fpe_selected_lag(self, result):
        assert result.selected["fpe"] == R_VAR["selected_lag"]["fpe"]
