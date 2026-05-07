from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

SIGNAL_LABELS = [
    "Adoption Lag (days)",
    "Brand Switch (0/1)",
    "Price Elasticity",
    "Attribute Priority",
    "Loyalty Retention",
]


def plot_entropy_weights(
    weights: np.ndarray,
    entropy: np.ndarray,
    feature_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    if feature_names is None:
        feature_names = SIGNAL_LABELS[: len(weights)]
        while len(feature_names) < len(weights):
            feature_names.append(f"Dim {len(feature_names) + 1}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Entropy Weighting — Signal Dimension Analysis", fontsize=13, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(
        feature_names,
        weights,
        color="#2e86c1",
        edgecolor="white",
        width=0.6,
    )
    ax.set_ylabel("Entropy Weight $w_j$", fontsize=10)
    ax.set_title("Assigned Entropy Weights per Signal Dimension", fontsize=10)
    ax.set_ylim(0, weights.max() * 1.25)
    for bar, w in zip(bars, weights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{w:.3f}",
            ha="center",
            fontsize=8,
        )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    ax2 = axes[1]
    ax2.bar(
        feature_names,
        entropy,
        color="#aed6f1",
        edgecolor="white",
        width=0.6,
    )
    ax2.axhline(y=entropy.mean(), color="#1a5276", linestyle="--", linewidth=1.2, label=f"Mean H = {entropy.mean():.3f}")
    ax2.set_ylabel("Normalized Entropy $H_j$", fontsize=10)
    ax2.set_title("Normalized Entropy per Signal Dimension", fontsize=10)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_explained_variance(
    evr: np.ndarray,
    n_components: int,
    variance_threshold: float = 0.72,
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    cumulative = np.cumsum(evr)
    components = np.arange(1, len(evr) + 1)

    ax.bar(components, evr * 100, color="#2e86c1", alpha=0.7, label="Individual EVR (%)")
    ax.plot(components, cumulative * 100, "o-", color="#1a5276", linewidth=2, markersize=5, label="Cumulative EVR (%)")
    ax.axhline(y=variance_threshold * 100, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Threshold ({variance_threshold*100:.0f}%)")
    ax.axvline(x=n_components, color="#27ae60", linestyle=":", linewidth=1.5, label=f"Selected k={n_components}")
    ax.set_xlabel("Principal Component", fontsize=10)
    ax.set_ylabel("Explained Variance Ratio (%)", fontsize=10)
    ax.set_title("Scree Plot — Behavioral Compression", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_silhouette(
    Z: np.ndarray,
    labels: np.ndarray,
    silhouette_samples: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    from sklearn.metrics import silhouette_score
    from caps.segmentation.maps import PP_COLORS

    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 10
    unique_labels = np.unique(labels)
    colors = list(PP_COLORS.values())

    for i, k in enumerate(unique_labels):
        ith_silhouette = silhouette_samples[labels == k]
        ith_silhouette.sort()
        size = ith_silhouette.shape[0]
        y_upper = y_lower + size
        color = colors[i % len(colors)]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_silhouette,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size, f"S{k}", fontsize=8)
        y_lower = y_upper + 10

    mean_sil = silhouette_score(Z, labels)
    ax.axvline(x=mean_sil, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean = {mean_sil:.3f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=10)
    ax.set_title("Silhouette Analysis — Cluster Validation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()