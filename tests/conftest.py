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