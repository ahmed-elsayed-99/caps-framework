from __future__ import annotations

import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform


class TwoStageClustering:
    def __init__(self, config):
        self.config = config
        self.optimal_k_: int = None
        self.labels_: np.ndarray = None
        self.centroids_: np.ndarray = None
        self.silhouette_score_: float = None
        self.silhouette_samples_: np.ndarray = None
        self.gap_statistics_: np.ndarray = None

    def fit(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        k_opt = self._gap_statistic(Z)
        self.optimal_k_ = k_opt

        agg = AgglomerativeClustering(n_clusters=k_opt, linkage="ward")
        agg_labels = agg.fit_predict(Z)
        init_centroids = np.array(
            [Z[agg_labels == k].mean(axis=0) for k in range(k_opt)]
        )

        kmeans = KMeans(
            n_clusters=k_opt,
            init=init_centroids,
            n_init=1,
            max_iter=500,
            random_state=self.config.random_state,
        )
        labels = kmeans.fit_predict(Z)
        centroids = kmeans.cluster_centers_

        sil = silhouette_score(Z, labels)
        sil_samples = silhouette_samples(Z, labels)

        self.labels_ = labels
        self.centroids_ = centroids
        self.silhouette_score_ = sil
        self.silhouette_samples_ = sil_samples

        return labels, centroids, sil

    def _gap_statistic(
        self, Z: np.ndarray, B: int = 10, k_max: int = None
    ) -> int:
        if k_max is None:
            k_max = min(self.config.max_clusters, len(Z) // 10)
        k_max = max(k_max, 2)
        rng = np.random.default_rng(self.config.random_state)

        def _inertia(data, k):
            km = KMeans(
                n_clusters=k,
                n_init=5,
                random_state=self.config.random_state,
            )
            km.fit(data)
            return km.inertia_

        Z_min = Z.min(axis=0)
        Z_max = Z.max(axis=0)

        gaps = np.zeros(k_max)
        gap_stds = np.zeros(k_max)

        for k in range(1, k_max + 1):
            W_k = np.log(_inertia(Z, k) + 1e-12)
            W_kb_list = []
            for _ in range(B):
                Z_rand = rng.uniform(low=Z_min, high=Z_max, size=Z.shape)
                W_kb_list.append(np.log(_inertia(Z_rand, k) + 1e-12))
            W_kb = np.array(W_kb_list)
            gaps[k - 1] = W_kb.mean() - W_k
            gap_stds[k - 1] = W_kb.std() * np.sqrt(1 + 1.0 / B)

        self.gap_statistics_ = gaps

        for k in range(1, k_max):
            if gaps[k - 1] >= gaps[k] - gap_stds[k]:
                return k

        return k_max

    def cluster_summary(self) -> dict:
        if self.labels_ is None:
            raise RuntimeError("fit must be called before cluster_summary.")
        unique = np.unique(self.labels_)
        return {
            int(k): {
                "size": int((self.labels_ == k).sum()),
                "centroid": self.centroids_[k].tolist(),
                "mean_silhouette": float(
                    self.silhouette_samples_[self.labels_ == k].mean()
                ),
            }
            for k in unique
        }