from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from caps.acquisition.signal_matrix import SignalMatrix
from caps.acquisition.validators import SignalValidator
from caps.compression.entropy_weighting import EntropyWeighter
from caps.compression.dimensionality import BehavioralCompressor
from caps.compression.clustering import TwoStageClustering
from caps.refinement.alignment import AlignmentValidator
from caps.segmentation.profiles import SegmentProfiler
from caps.segmentation.maps import SegmentMap
from caps.utils.metrics import PipelineMetrics
from caps.visualization.report import PipelineReport

logger = logging.getLogger(__name__)


@dataclass
class CAPSConfig:
    observation_window: int = 90
    variance_threshold: float = 0.72
    min_silhouette: float = 0.50
    min_alignment_score: float = 0.75
    min_kappa: float = 0.70
    n_qualitative_per_cluster: int = 8
    qualitative_floor: int = 8
    random_state: int = 42
    n_jobs: int = -1
    entropy_discretization_bins: int = 10
    max_clusters: int = 12
    switching_threshold: float = 0.30


@dataclass
class CAPSResults:
    config: CAPSConfig
    signal_matrix: np.ndarray
    weighted_matrix: np.ndarray
    entropy_weights: np.ndarray
    explained_variance_ratio: np.ndarray
    reduced_matrix: np.ndarray
    n_components: int
    cluster_labels: np.ndarray
    silhouette_score: float
    segment_profiles: list[dict]
    alignment_score: Optional[float]
    metrics: PipelineMetrics
    _profiler: SegmentProfiler = field(repr=False, default=None)

    def segment_map(self, save_path: Optional[str] = None) -> None:
        sm = SegmentMap(self)
        sm.plot(save_path=save_path)

    def summary(self) -> None:
        report = PipelineReport(self)
        report.print_summary()

    def to_dataframe(self) -> pd.DataFrame:
        return self._profiler.to_dataframe()

    def export(self, path: str) -> None:
        self._profiler.export(path)


class CAPSPipeline:
    def __init__(self, **kwargs):
        self.config = CAPSConfig(**kwargs)
        self._validator = SignalValidator(self.config)
        self._weighter = EntropyWeighter(self.config)
        self._compressor = BehavioralCompressor(self.config)
        self._clusterer = TwoStageClustering(self.config)
        self._alignment = AlignmentValidator(self.config)
        self._profiler = SegmentProfiler(self.config)

    def fit(self, df: pd.DataFrame) -> CAPSResults:
        logger.info("CAPS Pipeline — Layer 1: Signal Acquisition")
        sm = SignalMatrix(self.config)
        X, consumer_ids = sm.build(df)
        self._validator.validate(X)

        logger.info("CAPS Pipeline — Layer 2: Behavioral Compression")
        weights, H = self._weighter.fit_transform(X)
        X_weighted = self._weighter.apply(X, weights)
        Z, components, evr, n_components = self._compressor.fit_transform(X_weighted)
        labels, centroids, sil = self._clusterer.fit(Z)

        if sil < self.config.min_silhouette:
            logger.warning(
                f"Silhouette score {sil:.3f} below threshold "
                f"{self.config.min_silhouette}. Profiles produced but "
                f"qualitative refinement is strongly recommended before use."
            )

        logger.info("CAPS Pipeline — Layer 3: Qualitative Refinement (alignment check)")
        alignment_score = self._alignment.compute(Z, labels, components)

        logger.info("CAPS Pipeline — Segment Profiling")
        profiles = self._profiler.build_profiles(
            X, Z, labels, centroids, consumer_ids, weights
        )

        metrics = PipelineMetrics(
            n_consumers=len(consumer_ids),
            n_signal_dimensions=X.shape[1],
            n_components=n_components,
            explained_variance=float(evr[:n_components].sum()),
            silhouette_score=sil,
            n_clusters=len(np.unique(labels)),
            alignment_score=alignment_score,
            entropy_weights=weights,
        )

        results = CAPSResults(
            config=self.config,
            signal_matrix=X,
            weighted_matrix=X_weighted,
            entropy_weights=weights,
            explained_variance_ratio=evr,
            reduced_matrix=Z,
            n_components=n_components,
            cluster_labels=labels,
            silhouette_score=sil,
            segment_profiles=profiles,
            alignment_score=alignment_score,
            metrics=metrics,
            _profiler=self._profiler,
        )

        return results