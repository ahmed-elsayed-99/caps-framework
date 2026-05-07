import pytest
import numpy as np
from caps import CAPSPipeline
from caps.acquisition.loaders import _generate_synthetic_data


@pytest.fixture
def pipeline():
    return CAPSPipeline(
        observation_window=90,
        variance_threshold=0.72,
        min_silhouette=0.40,
        random_state=42,
        max_clusters=6,
    )


@pytest.fixture
def data():
    return _generate_synthetic_data(n_consumers=400, random_state=42)


def test_full_pipeline_runs(pipeline, data):
    results = pipeline.fit(data)
    assert results is not None


def test_results_have_profiles(pipeline, data):
    results = pipeline.fit(data)
    assert len(results.segment_profiles) >= 2


def test_results_labels_cover_all_consumers(pipeline, data):
    results = pipeline.fit(data)
    assert len(results.cluster_labels) == results.metrics.n_consumers


def test_segment_shares_sum_to_one(pipeline, data):
    results = pipeline.fit(data)
    total_share = sum(p["share_of_sample"] for p in results.segment_profiles)
    assert abs(total_share - 1.0) < 1e-6


def test_to_dataframe_columns(pipeline, data):
    results = pipeline.fit(data)
    df = results.to_dataframe()
    assert "consumer_id" in df.columns
    assert "segment_name" in df.columns
    assert "purchasing_power_tier" in df.columns


def test_metrics_pass_thresholds(pipeline, data):
    results = pipeline.fit(data)
    config = pipeline.config
    thresholds = results.metrics.passes_thresholds(config)
    assert thresholds["explained_variance"]


def test_entropy_weights_sum_to_one(pipeline, data):
    results = pipeline.fit(data)
    assert abs(results.entropy_weights.sum() - 1.0) < 1e-9


def test_reduced_matrix_shape(pipeline, data):
    results = pipeline.fit(data)
    n = results.metrics.n_consumers
    k = results.n_components
    assert results.reduced_matrix.shape == (n, k)