from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
import math


INTERVIEW_DOMAINS = {
    "D1_launch_response": {
        "description": "Consumer account of the competitive launch event",
        "probe_questions": [
            "How did you first become aware of the new product?",
            "What was your initial reaction when you saw it in the market?",
            "What did you do first after you noticed it?",
            "How long did it take you to decide whether to try it?",
        ],
        "first_order_code_examples": [
            "tried it immediately",
            "waited for reviews",
            "noticed the price difference",
            "did not notice the launch",
            "asked someone else about it",
        ],
    },
    "D2_comparative_valuation": {
        "description": "Price threshold structure and attribute weighting",
        "probe_questions": [
            "How did you compare this product to what you were already buying?",
            "What would the price need to be for you to switch to it?",
            "What was the most important thing you looked at when comparing them?",
            "Did you feel the quality difference was worth the price difference?",
        ],
        "first_order_code_examples": [
            "it had to be cheaper to switch",
            "quality was the main thing",
            "I was not going to pay more",
            "price did not really matter",
            "I compared the ingredients or features carefully",
        ],
    },
    "D3_consumption_context": {
        "description": "Expenditure prioritization and financial flexibility framing",
        "probe_questions": [
            "How important is this category in your overall spending?",
            "If this product cost 20% more, would that change your decision?",
            "Do you ever feel you have to choose between this and something else?",
            "How do you think about value when buying in this category?",
        ],
        "first_order_code_examples": [
            "it is not a big decision for me",
            "I had to choose between this and something else",
            "I always look for the best deal",
            "I do not think much about price in this category",
            "I set a budget for this kind of thing",
        ],
    },
}

PP_CONSTRUCT_INDICATORS = {
    "high": [
        "price did not really matter",
        "quality was the main thing",
        "tried it immediately",
        "it is not a big decision for me",
        "I do not think much about price in this category",
    ],
    "mid": [
        "I compared the ingredients or features carefully",
        "waited for reviews",
        "I set a budget for this kind of thing",
        "I was not going to pay more",
    ],
    "low": [
        "it had to be cheaper to switch",
        "I always look for the best deal",
        "I had to choose between this and something else",
        "I was not going to pay more",
        "did not notice the launch",
    ],
}


class QualitativeRefinement:
    def __init__(self, config):
        self.config = config
        self.coded_transcripts_: list[dict] = []
        self.cluster_profiles_: dict = {}

    def compute_sample_size(self, cluster_size: int) -> int:
        n_q = math.ceil(0.08 * cluster_size)
        return max(n_q, self.config.qualitative_floor)

    def select_participants(
        self,
        Z: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        cluster_id: int,
    ) -> np.ndarray:
        mask = labels == cluster_id
        cluster_Z = Z[mask]
        cluster_indices = np.where(mask)[0]
        centroid = centroids[cluster_id]
        dists = np.linalg.norm(cluster_Z - centroid, axis=1)
        n_q = self.compute_sample_size(cluster_Z.shape[0])
        n_core = max(1, n_q // 2)
        n_boundary = n_q - n_core

        core_idx = np.argsort(dists)[:n_core]
        boundary_idx = np.argsort(dists)[-n_boundary:]
        selected = np.union1d(core_idx, boundary_idx)
        return cluster_indices[selected]

    def code_transcript(
        self,
        transcript: str,
        consumer_id: str,
        cluster_id: int,
        rater_id: str = "R1",
    ) -> dict:
        transcript_lower = transcript.lower()
        codes = []
        pp_signals = {"high": 0, "mid": 0, "low": 0}

        for domain_key, domain in INTERVIEW_DOMAINS.items():
            domain_codes = []
            for code in domain["first_order_code_examples"]:
                if code.lower() in transcript_lower:
                    domain_codes.append(code)
            codes.append({"domain": domain_key, "first_order_codes": domain_codes})

        for level, indicators in PP_CONSTRUCT_INDICATORS.items():
            for ind in indicators:
                if ind.lower() in transcript_lower:
                    pp_signals[level] += 1

        pp_label = max(pp_signals, key=pp_signals.get)

        coded = {
            "consumer_id": consumer_id,
            "cluster_id": cluster_id,
            "rater_id": rater_id,
            "codes": codes,
            "pp_signal_counts": pp_signals,
            "pp_label": pp_label,
            "raw_transcript_length": len(transcript),
        }
        self.coded_transcripts_.append(coded)
        return coded

    def compute_kappa(
        self, ratings_r1: list[str], ratings_r2: list[str]
    ) -> float:
        assert len(ratings_r1) == len(ratings_r2)
        n = len(ratings_r1)
        categories = list(set(ratings_r1 + ratings_r2))
        observed_agreement = sum(
            1 for a, b in zip(ratings_r1, ratings_r2) if a == b
        ) / n

        expected = 0.0
        for cat in categories:
            p1 = ratings_r1.count(cat) / n
            p2 = ratings_r2.count(cat) / n
            expected += p1 * p2

        if abs(1 - expected) < 1e-10:
            return 1.0
        kappa = (observed_agreement - expected) / (1 - expected)
        return float(kappa)

    def cluster_pp_profile(self, cluster_id: int) -> dict:
        cluster_codes = [
            c for c in self.coded_transcripts_ if c["cluster_id"] == cluster_id
        ]
        if not cluster_codes:
            return {"cluster_id": cluster_id, "pp_label": "unknown", "confidence": 0.0}

        label_counts = {"high": 0, "mid": 0, "low": 0}
        for c in cluster_codes:
            label_counts[c["pp_label"]] += 1

        total = sum(label_counts.values())
        dominant = max(label_counts, key=label_counts.get)
        confidence = label_counts[dominant] / total if total > 0 else 0.0

        return {
            "cluster_id": cluster_id,
            "pp_label": dominant,
            "label_distribution": {k: v / total for k, v in label_counts.items()},
            "confidence": confidence,
            "n_coded": len(cluster_codes),
        }