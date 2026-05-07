from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from typing import Optional

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

PP_LABELS = {
    "high": "High Purchasing Power",
    "mid": "Mid Purchasing Power",
    "low": "Low Purchasing Power",
    "unknown": "Unclassified",
}


class SegmentMapPlotter:
    def __init__(
        self,
        Z: np.ndarray,
        labels: np.ndarray,
        profiles: list[dict],
        config=None,
    ):
        self.Z = Z
        self.labels = labels
        self.profiles = profiles
        self.profile_map = {p["cluster_id"]: p for p in profiles}
        self.config = config

    def plot_2d(
        self,
        pc_x: int = 0,
        pc_y: int = 1,
        figsize: tuple = (10, 7),
        save_path: Optional[str] = None,
        show_ellipses: bool = True,
        show_centroids: bool = True,
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("white")

        for k in np.unique(self.labels):
            p = self.profile_map[k]
            mask = self.labels == k
            tier = p["purchasing_power_tier"]
            color = PP_COLORS.get(tier, PP_COLORS["unknown"])
            marker = PP_MARKERS.get(tier, "D")

            ax.scatter(
                self.Z[mask, pc_x],
                self.Z[mask, pc_y],
                c=color,
                marker=marker,
                alpha=0.55,
                s=35,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )

            if show_ellipses and mask.sum() >= 3:
                self._draw_confidence_ellipse(
                    ax,
                    self.Z[mask, pc_x],
                    self.Z[mask, pc_y],
                    color=color,
                    alpha=0.12,
                )

            if show_centroids:
                cx = self.Z[mask, pc_x].mean()
                cy = self.Z[mask, pc_y].mean()
                ax.scatter(cx, cy, c=color, s=120, marker="X",
                           edgecolors="white", linewidths=1.2, zorder=5)
                ax.annotate(
                    f"  S{k}: {p['segment_name']}\n  PP={tier.upper()}\n  n={p['n_consumers']}",
                    (cx, cy),
                    fontsize=6.5,
                    color="#2c3e50",
                    zorder=6,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.85,
                        linewidth=0.8,
                    ),
                )

        legend_elements = []
        for tier in ["high", "mid", "low"]:
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker=PP_MARKERS[tier],
                    color="w",
                    markerfacecolor=PP_COLORS[tier],
                    markersize=9,
                    label=PP_LABELS[tier],
                )
            )
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="X",
                color="w",
                markerfacecolor="#2c3e50",
                markersize=9,
                label="Cluster Centroid",
            )
        )
        ax.legend(handles=legend_elements, fontsize=8, loc="best",
                  framealpha=0.9, edgecolor="#cccccc")

        ax.set_xlabel(
            f"PC{pc_x+1} — Primary Purchasing Power Axis", fontsize=10
        )
        ax.set_ylabel(
            f"PC{pc_y+1} — Secondary Behavioral Axis", fontsize=10
        )
        ax.set_title(
            "CAPS Segment Map — Behavioral Reduced Space",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_3d(
        self,
        figsize: tuple = (11, 8),
        save_path: Optional[str] = None,
    ) -> None:
        if self.Z.shape[1] < 3:
            raise ValueError(
                "3D segment map requires at least 3 principal components. "
                f"Current Z has {self.Z.shape[1]} columns."
            )
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        for k in np.unique(self.labels):
            p = self.profile_map[k]
            mask = self.labels == k
            tier = p["purchasing_power_tier"]
            color = PP_COLORS.get(tier, PP_COLORS["unknown"])
            marker = PP_MARKERS.get(tier, "D")

            ax.scatter(
                self.Z[mask, 0],
                self.Z[mask, 1],
                self.Z[mask, 2],
                c=color,
                marker=marker,
                alpha=0.5,
                s=30,
                label=f"S{k}: {p['segment_name']} [{tier.upper()}]",
            )
            cx = self.Z[mask, 0].mean()
            cy = self.Z[mask, 1].mean()
            cz = self.Z[mask, 2].mean()
            ax.scatter(cx, cy, cz, c=color, s=150, marker="X",
                       edgecolors="white", linewidths=1.0, zorder=5)

        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.set_zlabel("PC3", fontsize=9)
        ax.set_title(
            "CAPS 3D Segment Map",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=7, loc="best")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_radar(
        self,
        feature_names: Optional[list[str]] = None,
        figsize: tuple = (10, 8),
        save_path: Optional[str] = None,
    ) -> None:
        profiles_with_means = [
            p for p in self.profiles if p.get("signal_means")
        ]
        if not profiles_with_means:
            raise ValueError("No signal_means found in profiles.")

        first_means = profiles_with_means[0]["signal_means"]
        dims = list(first_means.keys())
        if feature_names is None:
            feature_names = dims
        N = len(dims)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        ax.set_facecolor("#f8f9fa")

        all_values = []
        for p in profiles_with_means:
            vals = [p["signal_means"].get(d, 0.0) for d in dims]
            all_values.extend(vals)
        v_min = min(all_values)
        v_max = max(all_values)
        v_range = v_max - v_min if v_max != v_min else 1.0

        for p in profiles_with_means:
            tier = p["purchasing_power_tier"]
            color = PP_COLORS.get(tier, PP_COLORS["unknown"])
            raw_vals = [p["signal_means"].get(d, 0.0) for d in dims]
            norm_vals = [(v - v_min) / v_range for v in raw_vals]
            norm_vals += norm_vals[:1]
            ax.plot(angles, norm_vals, color=color, linewidth=2, linestyle="solid")
            ax.fill(angles, norm_vals, color=color, alpha=0.15)
            ax.annotate(
                f"S{p['cluster_id']}",
                xy=(angles[0], norm_vals[0]),
                fontsize=8,
                color=color,
                fontweight="bold",
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, fontsize=8)
        ax.set_title(
            "CAPS Segment Radar Chart — Normalized Signal Means",
            fontsize=11,
            fontweight="bold",
            pad=20,
        )

        legend_elements = [
            Line2D([0], [0], color=PP_COLORS[t], linewidth=2,
                   label=f"{PP_LABELS[t]}")
            for t in ["high", "mid", "low"]
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper right",
                  bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_heatmap(
        self,
        feature_names: Optional[list[str]] = None,
        figsize: tuple = (10, 5),
        save_path: Optional[str] = None,
    ) -> None:
        import seaborn as sns

        profiles_with_means = [
            p for p in self.profiles if p.get("signal_means")
        ]
        dims = list(profiles_with_means[0]["signal_means"].keys())
        if feature_names is None:
            feature_names = dims

        matrix = np.array([
            [p["signal_means"].get(d, 0.0) for d in dims]
            for p in profiles_with_means
        ])
        row_labels = [
            f"S{p['cluster_id']}: {p['segment_name']}\n[{p['purchasing_power_tier'].upper()}]"
            for p in profiles_with_means
        ]

        matrix_norm = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-9)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            matrix_norm,
            ax=ax,
            xticklabels=feature_names,
            yticklabels=row_labels,
            cmap="RdBu_r",
            center=0,
            annot=matrix.round(3),
            fmt=".3f",
            linewidths=0.5,
            linecolor="#e0e0e0",
            cbar_kws={"label": "Z-Score (normalized across segments)"},
        )
        ax.set_title(
            "CAPS Signal Profile Heatmap — Segment vs. Signal Dimension",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _draw_confidence_ellipse(
        ax, x: np.ndarray, y: np.ndarray, color: str, alpha: float = 0.15
    ) -> None:
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms

        if len(x) < 3:
            return
        cov = np.cov(x, y)
        if cov.shape != (2, 2):
            return
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if np.any(eigenvalues <= 0):
            return
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues) * 2.0

        ellipse = Ellipse(
            xy=(np.mean(x), np.mean(y)),
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=1.2,
            linestyle="--",
            zorder=2,
        )
        ax.add_patch(ellipse)