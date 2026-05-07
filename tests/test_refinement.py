import pytest
import numpy as np
from caps.refinement.qualitative import QualitativeRefinement, PP_CONSTRUCT_INDICATORS
from caps.refinement.alignment import AlignmentValidator


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
def qualitative(config):
    return QualitativeRefinement(config)


@pytest.fixture
def alignment(config):
    return AlignmentValidator(config)


def test_sample_size_floor(qualitative):
    assert qualitative.compute_sample_size(10) == 8


def test_sample_size_scales(qualitative):
    assert qualitative.compute_sample_size(200) == max(8, int(0.08 * 200))


def test_select_participants_returns_indices(qualitative):
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((100, 2))
    labels = np.array([0] * 60 + [1] * 40)
    centroids = np.array([
        Z[:60].mean(axis=0),
        Z[60:].mean(axis=0),
    ])
    idx = qualitative.select_participants(Z, labels, centroids, cluster_id=0)
    assert len(idx) >= 8
    assert all(labels[i] == 0 for i in idx)


def test_code_transcript_high_pp(qualitative):
    transcript = (
        "price did not really matter to me. "
        "quality was the main thing. "
        "tried it immediately when I saw it. "
        "it is not a big decision for me financially."
    )
    result = qualitative.code_transcript(transcript, "C001", cluster_id=0)
    assert result["pp_label"] == "high"
    assert result["consumer_id"] == "C001"
    assert result["cluster_id"] == 0


def test_code_transcript_low_pp(qualitative):
    transcript = (
        "it had to be cheaper to switch. "
        "I always look for the best deal. "
        "I had to choose between this and something else."
    )
    result = qualitative.code_transcript(transcript, "C002", cluster_id=1)
    assert result["pp_label"] == "low"


def test_code_transcript_appends_to_list(qualitative):
    transcript = "quality was the main thing."
    qualitative.code_transcript(transcript, "C003", cluster_id=0)
    qualitative.code_transcript(transcript, "C004", cluster_id=0)
    assert len(qualitative.coded_transcripts_) == 2


def test_kappa_perfect_agreement(qualitative):
    r1 = ["high", "mid", "low", "high", "low"]
    r2 = ["high", "mid", "low", "high", "low"]
    kappa = qualitative.compute_kappa(r1, r2)
    assert abs(kappa - 1.0) < 1e-9


def test_kappa_zero_agreement(qualitative):
    r1 = ["high", "high", "high"]
    r2 = ["low", "low", "low"]
    kappa = qualitative.compute_kappa(r1, r2)
    assert kappa <= 0.0


def test_kappa_threshold(qualitative):
    r1 = ["high", "mid", "low", "high", "mid", "low", "high", "mid"]
    r2 = ["high", "mid", "low", "high", "mid", "low", "mid", "high"]
    kappa = qualitative.compute_kappa(r1, r2)
    assert isinstance(kappa, float)


def test_cluster_pp_profile_empty(qualitative):
    profile = qualitative.cluster_pp_profile(cluster_id=99)
    assert profile["pp_label"] == "unknown"
    assert profile["confidence"] == 0.0


def test_cluster_pp_profile_dominant(qualitative):
    transcripts_data = [
        "price did not really matter. quality was the main thing.",
        "price did not really matter. tried it immediately.",
        "it had to be cheaper to switch.",
    ]
    for i, t in enumerate(transcripts_data):
        qualitative.code_transcript(t, f"C{i:03d}", cluster_id=0)
    profile = qualitative.cluster_pp_profile(cluster_id=0)
    assert profile["pp_label"] in ["high", "mid", "low"]
    assert 0.0 <= profile["confidence"] <= 1.0
    assert profile["n_coded"] == 3


def test_alignment_compute_returns_float(alignment):
    rng = np.random.default_rng(42)
    Z = np.vstack([
        rng.normal([4, 0], 0.3, (50, 2)),
        rng.normal([0, 0], 0.3, (50, 2)),
        rng.normal([-4, 0], 0.3, (50, 2)),
    ])
    labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
    components = np.array([[1.0, 0.0], [0.0, 1.0]])
    score = alignment.compute(Z, labels, components)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_alignment_validates_qualitative(alignment):
    pc1_rank = np.array([0, 1, 2, 3])
    qual_rank = np.array([0, 1, 2, 3])
    score = alignment.validate_qualitative(pc1_rank, qual_rank)
    assert abs(score - 1.0) < 1e-9


def test_alignment_detects_misalignment(alignment):
    pc1_rank = np.array([0, 1, 2, 3])
    qual_rank = np.array([3, 2, 1, 0])
    score = alignment.validate_qualitative(pc1_rank, qual_rank)
    assert score < 0


def test_alignment_is_aligned_flag(alignment):
    pc1_rank = np.array([0, 1, 2, 3])
    qual_rank = np.array([0, 1, 2, 3])
    alignment.validate_qualitative(pc1_rank, qual_rank)
    assert alignment.is_aligned_ is True


def test_alignment_not_aligned_flag(alignment):
    pc1_rank = np.array([0, 1, 2, 3])
    qual_rank = np.array([3, 2, 1, 0])
    alignment.validate_qualitative(pc1_rank, qual_rank)
    assert alignment.is_aligned_ is False