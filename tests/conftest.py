from pathlib import Path as _Path

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def df_ols() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "income": np.random.lognormal(mean=4.5, sigma=0.6, size=n),
        "education": np.random.poisson(lam=14, size=n).clip(8, 22).astype(float),
        "age": np.random.uniform(22, 65, size=n),
        "female": np.random.binomial(1, 0.48, size=n).astype(float),
        "province": np.random.choice(["north", "central", "south"], size=n),
    })


@pytest.fixture
def df_oaxaca() -> pd.DataFrame:
    np.random.seed(99)
    n = 400
    age = np.random.uniform(22, 65, size=n)
    education = np.random.poisson(lam=13, size=n).clip(8, 16).astype(float)
    female = np.random.binomial(1, 0.5, size=n)
    gender_gap = 0.3 * female
    income = np.exp(
        1.5 + 0.08 * education + 0.02 * age - gender_gap + np.random.normal(0, 0.4, size=n)
    )
    return pd.DataFrame({
        "income": income,
        "education": education,
        "age": age,
        "female": female.astype(float),
    })


@pytest.fixture
def df_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["income", "education", "age"])


@pytest.fixture
def df_missing_col() -> pd.DataFrame:
    return pd.DataFrame({
        "income": [10, 20, 30],
        "age": [25, 30, 35],
    })


@pytest.fixture
def df_categorical() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "income": np.random.lognormal(mean=4.5, sigma=0.6, size=n),
        "education": np.random.poisson(lam=14, size=n).clip(8, 22).astype(float),
        "region": np.random.choice(["north", "central", "south"], size=n),
    })


@pytest.fixture
def df_collinear() -> pd.DataFrame:
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "y": np.random.uniform(0, 10, n),
        "x": np.random.uniform(0, 5, n),
    })
    df["x_dup"] = df["x"]
    return df


def _make_panel(n_unit: int = 60, n_time: int = 6, seed: int = 7) -> pd.DataFrame:
    """Balanced panel with entity and time fixed effects plus noise."""
    np.random.seed(seed)
    n = n_unit * n_time
    entity = np.repeat(np.arange(n_unit), n_time)
    time = np.tile(np.arange(n_time), n_unit)
    alpha = np.random.normal(0, 2, n_unit)
    beta_t = np.random.normal(0, 1.5, n_time)
    x = np.random.normal(0, 1, n)
    z = np.random.normal(0, 1, n)
    y = 1.5 * x - 0.7 * z + alpha[entity] + beta_t[time] + np.random.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "entity": entity, "time": time})


@pytest.fixture
def df_panel() -> pd.DataFrame:
    return _make_panel()


@pytest.fixture
def df_panel_unbalanced() -> pd.DataFrame:
    df = _make_panel(n_unit=60, n_time=6, seed=11)
    # Drop ~15% of rows to make the panel unbalanced.
    rng = np.random.default_rng(11)
    drop = rng.choice(df.index, size=int(0.15 * len(df)), replace=False)
    return df.drop(index=drop).reset_index(drop=True)


@pytest.fixture
def df_panel_single_entity() -> pd.DataFrame:
    df = _make_panel(n_unit=1, n_time=12, seed=5)
    return df


@pytest.fixture
def df_grunfeld() -> pd.DataFrame:
    from statsmodels.datasets import grunfeld

    g = grunfeld.load_pandas().data
    g = g.rename(columns={"firm": "firm", "year": "year"})
    g["firm"] = g["firm"].astype("category").cat.codes
    g["year"] = g["year"].astype(int)
    return g[["invest", "value", "capital", "firm", "year"]]


@pytest.fixture
def df_panel_dup_index() -> pd.DataFrame:
    df = _make_panel(n_unit=20, n_time=4, seed=9)
    # Duplicate one (entity, time) pair.
    dup_row = df.iloc[[5]].copy()
    return pd.concat([df, dup_row], ignore_index=True)


# ── CSV-backed Stata-parity fixtures ────────────────────────────────────────
# Relocated here from tests/stata/conftest.py (Dec 2024) to work around an
# order-dependent fixture-resolution failure (pytest ~9.1.1, Win, editable
# install) whose root cause was not fully identified.
#
# Trigger pattern: two test files importing open_econs.models.causal.rdd at
# module scope in the same session, combined with tests/stata/ files that
# depended on CSV-backed fixtures defined in tests/stata/conftest.py.  The
# fixture chain for tests/stata/*.py would sometimes bind to tests/conftest.py
# only, making the CSV-backed fixtures invisible.
#
# Candidate mechanisms ruled out before relocating:
#   · bare conftest-name collision (tests/*/__init__.py present;
#     confirmed distinct module names via __name__ tracing)
#   · sys.path insertion order (–import-mode=importlib had no effect)
#   · rddensity/rdrobust/lpdensity pytest plugin (none exist in env)
#   · stale duplicate open-econs installation (single editable install)
#   · Stata subprocess during collection (bug reproduced with STATA_EXE
#     disabled; module-level read_stata deferred to fixture)
#   · duplicate fixture definitions in sibling conftest (removing them
#     had no effect)
#
# Root cause suspected but not confirmed: import of open_econs.models.causal.rdd
# at module scope across two test files (one in tests/, one in tests/stata/)
# alters pytest's conftest-chain binding for the tests/stata/ directory under
# importlib/prepend modes in a way specific to pytest 9.x on Windows.
# Moving these four CSV-backed fixtures to the parent conftest makes them
# always visible regardless of collection order/import timing.

_STATA_FIXTURES = _Path(__file__).resolve().parent / "stata" / "fixtures" / "inputs"

def _load_stata_csv(name: str) -> pd.DataFrame:
    p = _STATA_FIXTURES / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Fixture not found: {p}")
    return pd.read_csv(p)

@pytest.fixture(scope="session")
def df_iv() -> pd.DataFrame:
    return _load_stata_csv("df_iv")


@pytest.fixture(scope="session")
def df_iv_cluster() -> pd.DataFrame:
    return _load_stata_csv("df_iv_cluster")


@pytest.fixture(scope="session")
def df_iv_panel() -> pd.DataFrame:
    return _load_stata_csv("df_iv_panel")

@pytest.fixture(scope="session")
def df_logit() -> pd.DataFrame:
    return _load_stata_csv("df_logit")

@pytest.fixture(scope="session")
def df_did() -> pd.DataFrame:
    return _load_stata_csv("df_did")

@pytest.fixture(scope="session")
def df_rdd() -> pd.DataFrame:
    return _load_stata_csv("df_rdd")