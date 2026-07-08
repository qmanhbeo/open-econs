"""Property-based adversarial tests using hypothesis.

Generate random-valid DataFrames and formulas, asserting that open-econs
either returns a sane result or raises a documented error — never an
unhandled traceback."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import open_econs as oe

# ── strategies ──────────────────────────────────────────────────────

n_strategy = st.integers(min_value=2, max_value=50)


@st.composite
def df_and_formula(draw, min_cols=2, max_cols=5):
    n = draw(n_strategy)
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    col_names = draw(
        st.lists(
            st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                    min_size=1, max_size=8).filter(lambda s: s.isidentifier()),
            min_size=n_cols,
            max_size=n_cols,
            unique=True,
        )
    )

    data = {}
    for i, name in enumerate(col_names):
        if i == 0:
            data[name] = draw(st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False), min_size=n, max_size=n))
        else:
            data[name] = draw(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=n, max_size=n))

    df = pd.DataFrame(data)

    predictors = col_names[1:]
    if not predictors:
        return df, None, None

    formula = f"{col_names[0]} ~ {' + '.join(predictors)}"
    return df, formula, col_names[0]


# ── OLS adversarial tests ───────────────────────────────────────────

class TestOLSRandom:
    @given(data=df_and_formula())
    @settings(max_examples=50, print_blob=False, deadline=None)
    def test_random_df(self, data):
        df, formula, _ = data
        if formula is None:
            return
        try:
            result = oe.ols(formula, data=df)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return
        except Exception as exc:
            pytest.fail(f"Unhandled exception for formula={formula}: {exc}")

        assert result.nobs == len(df)
        assert len(result.coefficients) > 0
        assert isinstance(result.tidy(), pd.DataFrame)
        assert isinstance(result.summary(), str)

    @settings(max_examples=30, print_blob=False)
    @given(
        n=n_strategy,
        has_nan=st.booleans(),
    )
    def test_collinear_or_nan(self, n, has_nan):
        df = pd.DataFrame({
            "y": np.random.uniform(0, 10, n),
            "x1": np.random.uniform(0, 10, n),
            "x2": np.full(n, 1.0),  # constant column — collinear with Intercept
        })
        if has_nan:
            df.iloc[0, 0] = np.nan
        try:
            oe.ols("y ~ x1 + x2", data=df)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return
        except Exception as e:
            pytest.fail(f"Unhandled: {e}")


# ── Oaxaca adversarial tests ───────────────────────────────────────

@st.composite
def oaxaca_df(draw):
    n = draw(st.integers(min_value=4, max_value=30))
    g0 = max(2, n // 2 - draw(st.integers(min_value=0, max_value=max(1, n // 4))))
    g1 = n - g0

    y0 = draw(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), min_size=g0, max_size=g0))
    y1 = draw(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), min_size=g1, max_size=g1))
    x0 = draw(st.lists(st.floats(min_value=0, max_value=50, allow_nan=False), min_size=g0, max_size=g0))
    x1 = draw(st.lists(st.floats(min_value=0, max_value=50, allow_nan=False), min_size=g1, max_size=g1))

    ys = y0 + y1
    xs = x0 + x1
    gs = [0.0] * g0 + [1.0] * g1

    df = pd.DataFrame({"y": ys, "x": xs, "g": gs})
    return df


@settings(max_examples=30, print_blob=False, deadline=None)
@given(df=oaxaca_df())
def test_oaxaca_random(df):
    try:
        d = oe.oaxaca("y ~ x + g", data=df, by="g")
    except (ValueError, RuntimeError):
        return
    except Exception as e:
        pytest.fail(f"Unhandled: {e}")

    assert d.total_gap >= 0  # swap=True ensures non-negative
    assert abs(d.explained + d.unexplained - d.total_gap) < 1e-8
    assert d.nobs == len(df)
    assert d.tidy() is not None