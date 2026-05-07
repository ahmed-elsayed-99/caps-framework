from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PipelineMetrics:
    n_consumers: int
    n_signal_dimensions: int
    n_components: int
    explained_variance: float
    silhouette_score: float
    n_clusters: int
    alignment_score: Optional[float]
    entropy_weights: np.ndarray

    def passes_thresholds(self, config) -> dict:
        return {
            "silhouette": self.silhouette_score >= config.min_silhouette,
            "explained_variance": self.explained_variance >= config.variance_threshold,
            "alignment": (
                self.alignment_score is None
                or self.alignment_score >= config.min_alignment_score
            ),
        }

    def to_dict(self) -> dict:
        return {
            "n_consumers": self.n_consumers,
            "n_signal_dimensions": self.n_signal_dimensions,
            "n_components": self.n_components,
            "explained_variance": self.explained_variance,
            "silhouette_score": self.silhouette_score,
            "n_clusters": self.n_clusters,
            "alignment_score": self.alignment_score,
            "entropy_weights": self.entropy_weights.tolist(),
        }