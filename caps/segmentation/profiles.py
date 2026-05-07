from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

SEGMENT_NAME_MAP = {
    "high": [
        "Premium Loyalists",
        "High-Capacity Adopters",
        "Quality-Anchor Segment",
    ],
    "mid": [
        "Flexible Mid-Tier",
        "Value-Conscious Switchers",
        "Selective Evaluators",
    ],
    "low": [
        "Price-Opportunist Segment",
        "Disengaged Low-Power",
        "Budget-Constrained Reactors",
    ],
}

SIGNAL_LABELS = [
    "adoption_lag_days",
    "brand_switch",
    "price_elasticity",
    "attribute_priority_score",
    "loyalty_retention_ratio",
]


class SegmentProfiler:
    def __init__(self, config):
        self.config = config
        self.profiles_: list[dict] = []
        self._df: Optional[pd.DataFrame] = None
        self._labels: Optional[np.ndarray] = None
        self._consumer_ids: Optional[np.ndarray] = None

    def build_profiles(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        consumer_ids: np.ndarray,
        weights: np.ndarray,
    ) -> list[dict]:
        self._labels = labels
        self._consumer_ids = consumer_ids
        unique_labels = np.unique(labels)
        profiles = []
        n_dims = min(X.shape[1], len(SIGNAL_LABELS))
        feature_names = SIGNAL_LABELS[:n_dims]
        if X.shape[1] > n_dims:
            for i in range(n_dims, X.shape[1]):
                feature_names.append(f"signal_dim_{i}")

        pc1_scores = [Z[labels == k, 0].mean() for k in unique_labels]
        pc1_ranks = np.argsort(np.argsort(pc1_scores))

        for idx, k in enumerate(unique_labels):
            mask = labels == k
            cluster_X = X[mask]
            cluster_Z = Z[mask]
            pc1_rank = int(pc1_ranks[idx])
            n_ranks = len(unique_labels)
            if pc1_rank >= n_ranks * 0.66:
                pp_tier = "high"
            elif pc1_rank >= n_ranks * 0.33:
                pp_tier = "mid"
            else:
                pp_tier = "low"

            name_pool = SEGMENT_NAME_MAP[pp_tier]
            segment_name = name_pool[k % len(name_pool)]

            signal_means = {
                feature_names[j]: float(cluster_X[:, j].mean())
                for j in range(len(feature_names))
            }
            signal_stds = {
                feature_names[j]: float(cluster_X[:, j].std())
                for j in range(len(feature_names))
            }

            profile = {
                "cluster_id": int(k),
                "segment_name": segment_name,
                "purchasing_power_tier": pp_tier,
                "n_consumers": int(mask.sum()),
                "share_of_sample": float(mask.sum() / len(labels)),
                "pc1_score_mean": float(cluster_Z[:, 0].mean()),
                "pc1_rank": pc1_rank,
                "centroid": centroids[k].tolist(),
                "signal_means": signal_means,
                "signal_stds": signal_stds,
                "entropy_weighted_signal": float(
                    np.dot(
                        [signal_means[f] for f in feature_names[:len(weights)]],
                        weights[:len(feature_names)],
                    )
                ),
                "qualitative_pp_label": None,
                "qualitative_confidence": None,
                "alignment_validated": False,
            }
            profiles.append(profile)

        self.profiles_ = profiles
        self._build_dataframe(X, labels, consumer_ids, feature_names)
        return profiles

    def _build_dataframe(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        consumer_ids: np.ndarray,
        feature_names: list[str],
    ) -> None:
        rows = []
        for i, cid in enumerate(consumer_ids):
            row = {"consumer_id": cid, "cluster_id": int(labels[i])}
            for j, fn in enumerate(feature_names):
                row[fn] = float(X[i, j])
            rows.append(row)
        self._df = pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("build_profiles must be called first.")
        profile_lookup = {p["cluster_id"]: p for p in self.profiles_}
        self._df["segment_name"] = self._df["cluster_id"].map(
            lambda k: profile_lookup[k]["segment_name"]
        )
        self._df["purchasing_power_tier"] = self._df["cluster_id"].map(
            lambda k: profile_lookup[k]["purchasing_power_tier"]
        )
        return self._df.copy()

    def export(self, path: str) -> None:
        df = self.to_dataframe()
        if path.endswith(".csv"):
            df.to_csv(path, index=False)
        elif path.endswith(".xlsx"):
            df.to_excel(path, index=False)
        else:
            df.to_csv(path, index=False)