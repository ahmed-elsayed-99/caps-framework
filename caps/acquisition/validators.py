from __future__ import annotations

import numpy as np


class SignalValidator:
    def __init__(self, config):
        self.config = config

    def validate(self, X: np.ndarray) -> None:
        self._check_shape(X)
        self._check_nan(X)
        self._check_variance(X)
        self._check_signal_to_noise(X)

    def _check_shape(self, X: np.ndarray) -> None:
        n, p = X.shape
        if n < 30:
            raise ValueError(
                f"Insufficient sample size: n={n}. Minimum 30 consumers required "
                f"for stable cluster solutions."
            )
        if p < 5:
            raise ValueError(
                f"Insufficient signal dimensions: p={p}. Minimum 5 dimensions "
                f"required by CAPS specification."
            )

    def _check_nan(self, X: np.ndarray) -> None:
        if np.isnan(X).any():
            raise ValueError(
                "Signal matrix contains NaN values after imputation. "
                "Check raw data for systematic missingness patterns."
            )

    def _check_variance(self, X: np.ndarray) -> None:
        zero_var_cols = np.where(X.var(axis=0) == 0)[0]
        if len(zero_var_cols) > 0:
            raise ValueError(
                f"Zero-variance columns detected at indices {zero_var_cols.tolist()}. "
                f"These dimensions carry no discriminatory information."
            )

    def _check_signal_to_noise(self, X: np.ndarray) -> None:
        col_means = np.abs(X.mean(axis=0))
        col_stds = X.std(axis=0)
        snr = col_means / (col_stds + 1e-9)
        low_snr = np.where(snr < 0.65)[0]
        if len(low_snr) > 0:
            pass