import pytest
import numpy as np
import pandas as pd
from caps.acquisition.signal_matrix import SignalMatrix
from caps.acquisition.validators import SignalValidator
from caps.acquisition.loaders import _generate_synthetic_data


class DummyConfig:
    observation_window = 90
    min_silhouette = 0.50
    variance_threshold = 0.72
    min_alignment_score = 0.75
    min_kappa = 0.70
    n_qualitative_per_cluster = 8
    qualitative_floor = 8
    random_state = 42
    n_jobs = -1
    entropy_discretization_bins = 10
    max_clusters = 12
    switching_threshold = 0.30


@pytest.fixture
def config():
    return DummyConfig()


@pytest.fixture
def sample_df():
    return _generate_synthetic_data(n_consumers=200, random_state=42)


def test_signal_matrix_build(config, sample_df):
    sm = SignalMatrix(config)
    X, consumer_ids = sm.build(sample_df)
    assert X.ndim == 2
    assert X.shape[0] == len(consumer_ids)
    assert X.shape[1] >= 5


def test_signal_matrix_filters_window(config, sample_df):
    config.observation_window = 30
    sm = SignalMatrix(config)
    X, consumer_ids = sm.build(sample_df)
    assert X.shape[0] <= 200


def test_signal_matrix_missing_column(config, sample_df):
    bad_df = sample_df.drop(columns=["price_elasticity"])
    sm = SignalMatrix(config)
    with pytest.raises(ValueError, match="Missing required columns"):
        sm.build(bad_df)


def test_validator_passes_valid(config, sample_df):
    sm = SignalMatrix(config)
    X, _ = sm.build(sample_df)
    v = SignalValidator(config)
    v.validate(X)


def test_validator_fails_small_sample(config):
    X = np.random.randn(10, 5)
    v = SignalValidator(config)
    with pytest.raises(ValueError, match="Insufficient sample size"):
        v.validate(X)


def test_validator_fails_few_dimensions(config):
    X = np.random.randn(100, 3)
    v = SignalValidator(config)
    with pytest.raises(ValueError, match="Insufficient signal dimensions"):
        v.validate(X)


def test_validator_fails_zero_variance(config):
    X = np.random.randn(100, 5)
    X[:, 2] = 0.0
    v = SignalValidator(config)
    with pytest.raises(ValueError, match="Zero-variance"):
        v.validate(X)


def test_synthetic_data_schema():
    df = _generate_synthetic_data(n_consumers=100)
    required = [
        "consumer_id",
        "adoption_lag_days",
        "brand_switch",
        "price_elasticity",
        "attribute_priority_score",
        "loyalty_retention_ratio",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_synthetic_data_value_ranges():
    df = _generate_synthetic_data(n_consumers=300)
    assert df["brand_switch"].isin([0, 1]).all()
    assert (df["loyalty_retention_ratio"] >= 0).all()
    assert (df["loyalty_retention_ratio"] <= 1).all()
    assert (df["attribute_priority_score"] >= 0).all()
    assert (df["attribute_priority_score"] <= 1).all()
    assert (df["adoption_lag_days"] >= 1).all()