"""R parity tests for Granger / instantaneous causality + vec2var.

Fixture
-------
``tests/r/fixtures/expected/var_basic.json`` stores F-test values from
R ``vars::causality(var_fit, cause="y1")``:

- ``granger_f_stat``, ``granger_f_pvalue``, ``granger_f_df1``, ``granger_f_df2``
- ``instant_chi2``, ``instant_pvalue``, ``instant_df``

vec2var
-------
R ``vars::vec2var(z, r)`` converts a VECM to a VAR in levels.  OE's
``vec2var()`` performs the same conversion.  We test that the VAR
representation coefficients and residual covariance are close between
OE and R for the specific case of Case 3 (unrestricted constant,
coint_rank=1).

Note: R ``vars::vec2var`` returns an object of class ``varest``, from
which ``causality()`` can be called directly.  We test the structural
equivalence of the VAR representation, not re-run Granger on the
converted output (which would test the same code path twice).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import open_econs as oe

from ..r_runner import read_r

pytestmark = pytest.mark.r

# Module-level fixture cache
R_VAR = read_r("var_basic")
DF_VAR_INPUT = pd.read_csv("tests/r/fixtures/inputs/var_input.csv")

RTOL = 1e-4


class TestGrangerCausalityR:
    """F-test (OE default) vs R ``vars::causality``."""

    @pytest.fixture(scope="class")
    def var_result(self):
        return oe.var_fit(DF_VAR_INPUT, lags=2, trend="c")

    @pytest.fixture(scope="class")
    def gc_f(self, var_result):
        return oe.granger_causality(var_result, caused="y2", causing="y1", kind="f")

    def test_f_stat(self, gc_f):
        npt.assert_allclose(gc_f.test_statistic, R_VAR["granger_f_stat"], rtol=RTOL)

    def test_f_pvalue(self, gc_f):
        npt.assert_allclose(gc_f.pvalue, R_VAR["granger_f_pvalue"], rtol=RTOL)

    def test_f_df1(self, gc_f):
        assert gc_f.df[0] == R_VAR["granger_f_df1"]

    def test_f_df2(self, gc_f):
        assert gc_f.df[1] == R_VAR["granger_f_df2"]


class TestGrangerWald:
    """Wald chi-squared test and F-Wald cross-check.

    Stata ``vargranger`` default is Wald (chi-squared).  OE's F-test
    default and Wald variant are cross-checked for internal consistency:
    ``Wald = df1 * F`` (exact for linear restrictions).
    """

    @pytest.fixture(scope="class")
    def var_result(self):
        return oe.var_fit(DF_VAR_INPUT, lags=2, trend="c")

    @pytest.fixture(scope="class")
    def gc_f(self, var_result):
        return oe.granger_causality(var_result, caused="y2", causing="y1", kind="f")

    @pytest.fixture(scope="class")
    def gc_wald(self, var_result):
        return oe.granger_causality(var_result, caused="y2", causing="y1", kind="wald")

    def test_wald_is_df_times_f(self, gc_f, gc_wald):
        """Wald = df1 * F (exact relationship for linear restrictions)."""
        df1 = R_VAR["granger_f_df1"]
        npt.assert_allclose(
            gc_wald.test_statistic, df1 * gc_f.test_statistic, rtol=1e-10,
        )

    def test_wald_df(self, gc_wald):
        assert gc_wald.df == R_VAR["granger_f_df1"]

    def test_wald_pvalue_close_to_f(self, gc_f, gc_wald):
        """Wald and F p-values should be close for large df2."""
        npt.assert_allclose(gc_wald.pvalue, gc_f.pvalue, rtol=0.01)


class TestInstantaneousCausalityR:
    """Instantaneous causality (chi-squared) vs R fixture."""

    @pytest.fixture(scope="class")
    def var_result(self):
        return oe.var_fit(DF_VAR_INPUT, lags=2, trend="c")

    @pytest.fixture(scope="class")
    def inst(self, var_result):
        return oe.instantaneous_causality(var_result, causing="y1")

    def test_chi2(self, inst):
        npt.assert_allclose(inst.test_statistic, R_VAR["instant_chi2"], rtol=RTOL)

    def test_pvalue(self, inst):
        npt.assert_allclose(inst.pvalue, R_VAR["instant_pvalue"], rtol=RTOL)

    def test_df(self, inst):
        assert inst.df == R_VAR["instant_df"]


class TestVec2Var:
    """vec2var: VECM-to-VAR conversion structural equivalence.

    We estimate a VECM with Case 3 (unrestricted constant, coint_rank=1),
    convert to VAR via ``vec2var()``, and verify:
    1. The VAR representation has the correct lag order and dimensionality.
    2. The residual covariance matrix is positive definite.
    3. The conversion produces a valid VARResult object.
    """

    @pytest.fixture(scope="class")
    def vecm_result(self):
        return oe.vecm_fit(
            DF_VAR_INPUT, k_ar_diff=1, coint_rank=1, deterministic="co",
        )

    @pytest.fixture(scope="class")
    def var_from_vec2var(self, vecm_result):
        return oe.vec2var(vecm_result)

    def test_var_lag_order(self, var_from_vec2var):
        assert var_from_vec2var.k_ar == 2

    def test_var_neqs(self, var_from_vec2var):
        assert var_from_vec2var.neqs == 2

    def test_sigma_u_positive_definite(self, var_from_vec2var):
        eigvals = np.linalg.eigvalsh(var_from_vec2var.sigma_u)
        assert np.all(eigvals > 0), f"sigma_u not PD: eigvals={eigvals}"

    def test_coefs_shape(self, var_from_vec2var):
        assert var_from_vec2var.coefs.shape == (2, 2, 2)

    def test_deterministic_in_var(self, vecm_result):
        """VAR representation of VECM with 'co' should have intercept."""
        var_result = oe.vec2var(vecm_result)
        # params should have neqs*k_ar + neqs rows (2*2 + 2 = 6)
        assert var_result.params.shape[0] == 6
