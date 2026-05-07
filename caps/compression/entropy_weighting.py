from __future__ import annotations

import numpy as np
from typing import Tuple


class EntropyWeighter:
    def __init__(self, config):
        self.config = config
        self.weights_: np.ndarray = None
        self.entropy_: np.ndarray = None
        self.divergence_: np.ndarray = None
        self._n_bins: int = config.entropy_discretization_bins

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n, p = X.shape
        H = np.zeros(p)

        for j in range(p):
            col = X[:, j]
            bins = np.histogram_bin_edges(col, bins=self._n_bins)
            digitized = np.digitize(col, bins[:-1]) - 1
            digitized = np.clip(digitized, 0, self._n_bins - 1)
            counts = np.bincount(digitized, minlength=self._n_bins)
            freq = counts / n
            freq = freq[freq > 0]
            H[j] = -np.sum(freq * np.log(freq + 1e-12)) / np.log(n + 1e-12)

        self.entropy_ = H
        d = 1.0 - H
        d = np.clip(d, 0.0, None)
        d_sum = d.sum()

        if d_sum < 1e-10:
            w = np.ones(p) / p
        else:
            w = d / d_sum

        self.weights_ = w
        self.divergence_ = d
        return w, H

    def apply(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        return X_std * weights[np.newaxis, :]

    def weight_table(self) -> dict:
        if self.weights_ is None:
            raise RuntimeError("fit_transform must be called before weight_table.")
        return {
            "entropy": self.entropy_.tolist(),
            "divergence": self.divergence_.tolist(),
            "weights": self.weights_.tolist(),
        }