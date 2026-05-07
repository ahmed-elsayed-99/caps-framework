from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

REQUIRED_COLUMNS = [
    "consumer_id",
    "adoption_lag_days",
    "brand_switch",
    "price_elasticity",
    "attribute_priority_score",
    "loyalty_retention_ratio",
]

OPTIONAL_COLUMNS = [
    "category_spend_share",
    "trial_frequency",
    "price_sensitivity_index",
]


class SignalMatrix:
    def __init__(self, config):
        self.config = config
        self._column_map = {
            "adoption_lag_days": 0,
            "brand_switch": 1,
            "price_elasticity": 2,
            "attribute_priority_score": 3,
            "loyalty_retention_ratio": 4,
        }

    def build(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = self._validate_columns(df)
        df = self._filter_window(df)
        df = self._impute(df)
        consumer_ids = df["consumer_id"].values
        feature_cols = [c for c in REQUIRED_COLUMNS if c != "consumer_id"]
        if "category_spend_share" in df.columns:
            feature_cols.append("category_spend_share")
        if "trial_frequency" in df.columns:
            feature_cols.append("trial_frequency")
        X = df[feature_cols].values.astype(np.float64)
        return X, consumer_ids

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Signal matrix build failed. Missing required columns: {missing}. "
                f"See data/README.md for schema documentation."
            )
        return df.copy()

    def _filter_window(self, df: pd.DataFrame) -> pd.DataFrame:
        if "adoption_lag_days" in df.columns:
            mask = df["adoption_lag_days"] <= self.config.observation_window
            n_excluded = (~mask).sum()
            if n_excluded > 0:
                pass
            df = df[mask].reset_index(drop=True)
        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        return df