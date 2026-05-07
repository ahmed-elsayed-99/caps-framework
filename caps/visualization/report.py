from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if TYPE_CHECKING:
    from caps.pipeline import CAPSResults

PP_TIER_STYLE = {
    "high": "bold blue",
    "mid": "bold cyan",
    "low": "bold white",
    "unknown": "dim",
}


class PipelineReport:
    def __init__(self, results: "CAPSResults"):
        self.results = results
        if RICH_AVAILABLE:
            self.console = Console()

    def print_summary(self) -> None:
        if RICH_AVAILABLE:
            self._rich_summary()
        else:
            self._plain_summary()

    def _rich_summary(self) -> None:
        m = self.results.metrics
        self.console.print()
        self.console.print(
            Panel.fit(
                "[bold]CAPS Framework — Pipeline Results[/bold]",
                border_style="blue",
            )
        )
        meta_table = Table(show_header=False, box=None, padding=(0, 2))
        meta_table.add_column("Key", style="bold")
        meta_table.add_column("Value")
        meta_table.add_row("Consumers", str(m.n_consumers))
        meta_table.add_row("Signal Dimensions", str(m.n_signal_dimensions))
        meta_table.add_row("Components Retained", str(m.n_components))
        meta_table.add_row("Explained Variance", f"{m.explained_variance:.3f}")
        meta_table.add_row("Silhouette Score", f"{m.silhouette_score:.3f}")
        meta_table.add_row("Clusters", str(m.n_clusters))
        if m.alignment_score is not None:
            meta_table.add_row("Alignment Score", f"{m.alignment_score:.3f}")
        self.console.print(meta_table)
        self.console.print()

        seg_table = Table(title="Segment Profiles", show_lines=True)
        seg_table.add_column("ID", style="dim")
        seg_table.add_column("Segment Name")
        seg_table.add_column("PP Tier")
        seg_table.add_column("n")
        seg_table.add_column("Share %")
        seg_table.add_column("PC1 Score")
        seg_table.add_column("Adoption Lag")
        seg_table.add_column("Price Elasticity")
        seg_table.add_column("Loyalty")

        for p in self.results.segment_profiles:
            tier = p["purchasing_power_tier"]
            style = PP_TIER_STYLE.get(tier, "")
            sm = p.get("signal_means", {})
            seg_table.add_row(
                str(p["cluster_id"]),
                p["segment_name"],
                Text(tier.upper(), style=style),
                str(p["n_consumers"]),
                f"{p['share_of_sample']*100:.1f}%",
                f"{p['pc1_score_mean']:.3f}",
                f"{sm.get('adoption_lag_days', float('nan')):.1f}",
                f"{sm.get('price_elasticity', float('nan')):.3f}",
                f"{sm.get('loyalty_retention_ratio', float('nan')):.3f}",
            )
        self.console.print(seg_table)
        self.console.print()

        w = m.entropy_weights
        w_table = Table(title="Entropy Weights", show_lines=False, box=None)
        w_table.add_column("Dimension")
        w_table.add_column("Weight")
        dim_names = [
            "Adoption Lag",
            "Brand Switch",
            "Price Elasticity",
            "Attr. Priority",
            "Loyalty Retention",
        ]
        for i, wi in enumerate(w):
            name = dim_names[i] if i < len(dim_names) else f"Dim {i}"
            w_table.add_row(name, f"{wi:.4f}")
        self.console.print(w_table)

    def _plain_summary(self) -> None:
        m = self.results.metrics
        print("\n" + "=" * 60)
        print("CAPS Framework — Pipeline Results")
        print("=" * 60)
        print(f"Consumers          : {m.n_consumers}")
        print(f"Signal Dimensions  : {m.n_signal_dimensions}")
        print(f"Components         : {m.n_components}")
        print(f"Explained Variance : {m.explained_variance:.3f}")
        print(f"Silhouette Score   : {m.silhouette_score:.3f}")
        print(f"Clusters           : {m.n_clusters}")
        if m.alignment_score is not None:
            print(f"Alignment Score    : {m.alignment_score:.3f}")
        print("-" * 60)
        for p in self.results.segment_profiles:
            print(
                f"  Segment {p['cluster_id']}: {p['segment_name']} "
                f"[{p['purchasing_power_tier'].upper()}] "
                f"n={p['n_consumers']} ({p['share_of_sample']*100:.1f}%)"
            )
        print("=" * 60 + "\n")