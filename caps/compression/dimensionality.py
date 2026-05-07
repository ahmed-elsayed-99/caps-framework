from __future__ import annotations

import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA


class BehavioralCompressor:
    def __init__(self, config):
        self.config = config
        self.pca_: PCA = None
        self.n_components_: int = None
        self.explained_variance_ratio_: np.ndarray = None

    def fit_transform(
        self, X_weighted: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        p = X_weighted.shape[1]
        pca_full = PCA(n_components=p, random_state=self.config.random_state)
        pca_full.fit(X_weighted)
        evr = pca_full.explained_variance_ratio_
        cumulative = np.cumsum(evr)
        n_components = int(np.searchsorted(cumulative, self.config.variance_threshold) + 1)
        n_components = max(2, min(n_components, p))

        pca_k = PCA(n_components=n_components, random_state=self.config.random_state)
        Z = pca_k.fit_transform(X_weighted)

        self.pca_ = pca_k
        self.n_components_ = n_components
        self.explained_variance_ratio_ = evr

        return Z, pca_k.components_, evr, n_components

    def transform(self, X_weighted: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("fit_transform must be called before transform.")
        return self.pca_.transform(X_weighted)

    def loadings_table(self, feature_names: list[str] = None) -> dict:
        if self.pca_ is None:
            raise RuntimeError("fit_transform must be called before loadings_table.")
        components = self.pca_.components_
        if feature_names is None:
            feature_names = [f"dim_{i}" for i in range(components.shape[1])]
        return {
            f"PC{i+1}": dict(zip(feature_names, components[i].tolist()))
            for i in range(components.shape[0])
        }