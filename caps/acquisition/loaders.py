from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import importlib.resources


def load_sample_data() -> pd.DataFrame:
    try:
        with importlib.resources.path("caps.data.raw", "sample_launch_event.csv") as p:
            return pd.read_csv(p)
    except Exception:
        return _generate_synthetic_data()


def load_from_csv(path: str, **kwargs) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(p, **kwargs)


def load_from_excel(path: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_excel(p, sheet_name=sheet_name, **kwargs)


def _generate_synthetic_data(
    n_consumers: int = 500,
    n_segments: int = 4,
    random_state: int = 42,
    observation_window: int = 90,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    segment_params = {
        0: {
            "name": "High-Power Adopters",
            "adoption_lag_mean": 5,
            "adoption_lag_std": 3,
            "switch_prob": 0.15,
            "elasticity_mean": -0.4,
            "elasticity_std": 0.15,
            "attr_priority_mean": 0.85,
            "attr_priority_std": 0.10,
            "loyalty_mean": 0.90,
            "loyalty_std": 0.08,
            "weight": 0.20,
        },
        1: {
            "name": "Flexible Mid-Tier",
            "adoption_lag_mean": 25,
            "adoption_lag_std": 10,
            "switch_prob": 0.45,
            "elasticity_mean": -1.1,
            "elasticity_std": 0.30,
            "attr_priority_mean": 0.55,
            "attr_priority_std": 0.15,
            "loyalty_mean": 0.65,
            "loyalty_std": 0.12,
            "weight": 0.35,
        },
        2: {
            "name": "Price-Opportunist",
            "adoption_lag_mean": 15,
            "adoption_lag_std": 8,
            "switch_prob": 0.80,
            "elasticity_mean": -2.2,
            "elasticity_std": 0.45,
            "attr_priority_mean": 0.30,
            "attr_priority_std": 0.12,
            "loyalty_mean": 0.35,
            "loyalty_std": 0.15,
            "weight": 0.30,
        },
        3: {
            "name": "Disengaged Low-Power",
            "adoption_lag_mean": 75,
            "adoption_lag_std": 12,
            "switch_prob": 0.60,
            "elasticity_mean": -3.1,
            "elasticity_std": 0.50,
            "attr_priority_mean": 0.20,
            "attr_priority_std": 0.10,
            "loyalty_mean": 0.25,
            "loyalty_std": 0.10,
            "weight": 0.15,
        },
    }

    weights = [p["weight"] for p in segment_params.values()]
    segment_assignments = rng.choice(
        list(segment_params.keys()),
        size=n_consumers,
        p=weights,
    )

    records = []
    for i, seg_id in enumerate(segment_assignments):
        p = segment_params[seg_id]

        adoption_lag = max(
            1,
            int(rng.normal(p["adoption_lag_mean"], p["adoption_lag_std"])),
        )
        adoption_lag = min(adoption_lag, observation_window)

        brand_switch = int(rng.random() < p["switch_prob"])

        elasticity = float(rng.normal(p["elasticity_mean"], p["elasticity_std"]))
        elasticity = np.clip(elasticity, -5.0, -0.05)

        attr_priority = float(
            rng.normal(p["attr_priority_mean"], p["attr_priority_std"])
        )
        attr_priority = np.clip(attr_priority, 0.0, 1.0)

        loyalty = float(rng.normal(p["loyalty_mean"], p["loyalty_std"]))
        loyalty = np.clip(loyalty, 0.0, 1.0)

        category_spend = float(rng.beta(2 + seg_id * 0.5, 5 - seg_id * 0.5))
        trial_freq = max(1, int(rng.poisson(3 - seg_id * 0.5)))

        records.append(
            {
                "consumer_id": f"C{i+1:05d}",
                "true_segment": p["name"],
                "adoption_lag_days": adoption_lag,
                "brand_switch": brand_switch,
                "price_elasticity": elasticity,
                "attribute_priority_score": attr_priority,
                "loyalty_retention_ratio": loyalty,
                "category_spend_share": category_spend,
                "trial_frequency": trial_freq,
            }
        )

    return pd.DataFrame(records)