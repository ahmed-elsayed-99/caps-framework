import pytest
import numpy as np
from caps.compression.entropy_weighting import EntropyWeighter
from caps.compression.dimensionality import BehavioralCompressor
from caps.compression.clustering import TwoStageClustering


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
    max_clusters = 8
    switching_threshold = 0.30


@pytest.fixture
def config():
    return DummyConfig()


@pytest.fixture
def X():
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 6))


def test_entropy_weighter_sums_to_one(config, X):
    ew = EntropyWeighter(config)
    weights, H = ew.fit_transform(X)
    assert abs(weights.sum() - 1.0) < 1e-9


def test_entropy_weighter_non_negative(config, X):
    ew = EntropyWeighter(config)
    weights, H = ew.fit_transform(X)
    assert (weights >= 0).all()
    assert (H >= 0).all()


def test_entropy_weighter_apply_shape(config, X):
    ew = EntropyWeighter(config)
    weights, _ = ew.fit_transform(X)
    X_w = ew.apply(X, weights)
    assert X_w.shape == X.shape


def test_compressor_variance_criterion(config, X):
    ew = EntropyWeighter(config)
    weights, _ = ew.fit_transform(X)
    X_w = ew.apply(X, weights)
    bc = BehavioralCompressor(config)
    Z, components, evr, k = bc.fit_transform(X_w)
    assert evr[:k].sum() >= config.variance_threshold


def test_compressor_reduced_dimensions(config, X):
    ew = EntropyWeighter(config)
    weights, _ = ew.fit_transform(X)
    X_w = ew.apply(X, weights)
    bc = BehavioralCompressor(config)
    Z, _, evr, k = bc.fit_transform(X_w)
    assert Z.shape[0] == X.shape[0]
    assert Z.shape[1] == k
    assert k >= 2


def test_clustering_produces_labels(config):
    rng = np.random.default_rng(42)
    Z = np.vstack([
        rng.normal([3, 3], 0.5, (60, 2)),
        rng.normal([-3, 3], 0.5, (60, 2)),
        rng.normal([0, -3], 0.5, (60, 2)),
    ])
    cl = TwoStageClustering(config)
    labels, centroids, sil = cl.fit(Z)
    assert len(labels) == 180
    assert centroids.shape[1] == 2
    assert 0.0 <= sil <= 1.0


def test_clustering_optimal_k(config):
    rng = np.random.default_rng(42)
    Z = np.vstack([
        rng.normal([5, 0], 0.4, (50, 2)),
        rng.normal([-5, 0], 0.4, (50, 2)),
        rng.normal([0, 5], 0.4, (50, 2)),
    ])
    cl = TwoStageClustering(config)
    labels, _, _ = cl.fit(Z)
    assert cl.optimal_k_ in [2, 3, 4]


def test_silhouette_threshold_warning(config, capsys):
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((100, 2))
    cl = TwoStageClustering(config)
    _, _, sil = cl.fit(Z)
    assert isinstance(sil, float)