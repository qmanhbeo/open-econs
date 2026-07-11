"""Tests for psm() — Abadie-Imbens (2006, 2012) 1:1 PSM with replacement."""

import numpy as np
import pandas as pd
import pytest

from open_econs.models.causal.psm import psm


FIXTURE = "tests/stata/fixtures/df_psm.csv"


@pytest.fixture
def df():
    return pd.read_csv(FIXTURE)


def test_psm_ate_matches_stata(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0)
    assert r.effect == pytest.approx(1.5333212, abs=1e-6)


def test_psm_se_nn2_matches_stata(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0, nn=2)
    assert r.se == pytest.approx(0.0765811, abs=1e-6)


def test_psm_se_nn5_matches_stata(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0, nn=5)
    assert r.se == pytest.approx(0.08283881, abs=1e-6)


def test_psm_se_nn10_matches_stata(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0, nn=10)
    assert r.se == pytest.approx(0.08351363, abs=1e-6)


def test_psm_result_shape(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0)
    assert r.n_matched == 500
    assert r.n_treated == 470
    assert r.n_control == 530
    assert isinstance(r.tidy(), pd.DataFrame)
    assert isinstance(r.summary(), str)


def test_psm_caliper_drops_outliers(df):
    r = psm(df, treatment="t", covariates=["x1", "x2"], caliper=0.01)
    assert r.n_matched < 500
    assert r.effect is not None


def test_psm_common_support(df):
    r1 = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0, common_support=False)
    r2 = psm(df, treatment="t", covariates=["x1", "x2"], caliper=1.0, common_support=True)
    assert r1.n_matched == 500
    assert r2.n_matched < r1.n_matched
