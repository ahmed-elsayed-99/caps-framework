from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def save_results_json(results, path: str) -> None:
    data = {
        "metrics": results.metrics.to_dict(),
        "segment_profiles": results.segment_profiles,
        "config": {
            k: v for k, v in results.config.__dict__.items()
            if not k.startswith("_")
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)


def load_results_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_segment_assignments(
    consumer_ids: np.ndarray,
    labels: np.ndarray,
    profiles: list[dict],
    path: str,
) -> None:
    profile_map = {p["cluster_id"]: p for p in profiles}
    rows = [
        {
            "consumer_id": cid,
            "cluster_id": int(labels[i]),
            "segment_name": profile_map[int(labels[i])]["segment_name"],
            "purchasing_power_tier": profile_map[int(labels[i])]["purchasing_power_tier"],
        }
        for i, cid in enumerate(consumer_ids)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)