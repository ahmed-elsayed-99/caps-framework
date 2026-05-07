from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from typing import Optional


class AlignmentValidator:
    def __init__(self, config):
        self.config = config
        self.alignment_score_: Optional[float] = None
        self.pc1_cluster_order_: Optional[np.ndarray] = None
        self.is_aligned_: Optional[bool] = None

    def compute(
        self,
        Z: np.ndarray,
        labels: np.ndarray,
        components: np.ndarray,
    ) -> float:
        pc1_loadings = components[0]
        pp_direction = self._infer_pp_direction(pc1_loadings)
        unique_labels = np.unique(labels)
        cluster_pc1_scores = np.array(
            [Z[labels == k, 0].mean() * pp_direction for k in unique_labels]
        )
        pc1_rank = np.argsort(np.argsort(cluster_pc1_scores))
        self.pc1_cluster_order_ = pc1_rank

        if len(unique_labels) < 3:
            self.alignment_score_ = 1.0
            self.is_aligned_ = True
            return 1.0

        monotone_score = self._monotone_consistency(cluster_pc1_scores)
        self.alignment_score_ = float(monotone_score)
        self.is_aligned_ = monotone_score >= self.config.min_alignment_score
        return self.alignment_score_

    def validate_qualitative(
        self,
        pc1_cluster_rank: np.ndarray,
        qualitative_pp_rank: np.ndarray,
    ) -> float:
        if len(pc1_cluster_rank) < 3:
            return 1.0
        rho, _ = spearmanr(pc1_cluster_rank, qualitative_pp_rank)
        self.alignment_score_ = float(rho)
        self.is_aligned_ = rho >= self.config.min_alignment_score
        return float(rho)

    def _infer_pp_direction(self, pc1_loadings: np.ndarray) -> float:
        dim_3_idx = min(2, len(pc1_loadings) - 1)
        return -1.0 if pc1_loadings[dim_3_idx] < 0 else 1.0

    def _monotone_consistency(self, scores: np.ndarray) -> float:
        n = len(scores)
        pairs = 0
        concordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                pairs += 1
                if scores[i] != scores[j]:
                    concordant += 1
        if pairs == 0:
            return 1.0
        return concordant / pairs