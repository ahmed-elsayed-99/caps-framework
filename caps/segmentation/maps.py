from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from caps.pipeline import CAPSResults

PP_COLORS = {
    "high": "#1a5276",
    "mid": "#2e86c1",
    "low": "#aed6f1",
    "unknown": "#99a3a4",
}

PP_MARKERS = {
    "high": "o",
    "mid": "s",
    "low": "^",
    "unknown": "D",
}


class SegmentMap:
    def __init__(self, results: "CAPSResults"):
        self.results = results

    def plot(self, save_path: Optional[str] = None, figsize: tuple = (12, 8)) -> None:
        Z = self.results.reduced_matrix
        labels = self.results.cluster_labels
        profiles = self.results.segment_profiles
        profile_map = {p["cluster_id"]: p for p in profiles}

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(
            "CAPS Segment Map — Behavioral Reduced Space",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )

        ax = axes[0]
        legend_handles = []
        for k in np.unique(labels):
            p = profile_map[k]
            mask = labels == k
            tier = p["purchasing_power_tier"]
            color = PP_COLORS.get(tier, PP_COLORS["unknown"])
            marker = PP_MARKERS.get(tier, "D")
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                c=color,
                marker=marker,
                alpha=0.65,
                s=40,
                label=f"[{tier.upper()}] {p['segment_name']} (n={p['n_consumers']})",
                edgecolors="white",
                linewidths=0.4,
            )
            cx, cy = p["centroid"][0], p["centroid"][1]
            ax.annotate(
                f"S{k}\n{p['purchasing_power_tier'].upper()}",
                (cx, cy),
                fontsize=7,
                ha="center",
                va="center",
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.85),
            )
            patch = mpatches.Patch(
                color=color,
                label=f"Segment {k}: {p['segment_name']}",
            )
            legend_handles.append(patch)

        ax.set_xlabel("PC1 (Primary Purchasing Power Axis)", fontsize=10)
        ax.set_ylabel("PC2 (Secondary Behavioral Axis)", fontsize=10)
        ax.set_title("Consumer Distribution in Reduced Space", fontsize=11)
        ax.legend(handles=legend_handles, fontsize=7, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax2 = axes[1]
        segment_names = [p["segment_name"] for p in profiles]
        shares = [p["share_of_sample"] * 100 for p in profiles]
        tiers = [p["purchasing_power_tier"] for p in profiles]
        colors = [PP_COLORS.get(t, PP_COLORS["unknown"]) for t in tiers]
        bars = ax2.barh(segment_names, shares, color=colors, edgecolor="white", height=0.6)
        ax2.set_xlabel("Share of Sample (%)", fontsize=10)
        ax2.set_title("Segment Size Distribution", fontsize=11)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for bar, val in zip(bars, shares):
            ax2.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()