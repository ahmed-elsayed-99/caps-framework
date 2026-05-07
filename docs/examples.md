# Examples

## Installation

```bash
pip install caps-framework
```

## Minimal Example

```python
from caps import CAPSPipeline
from caps.acquisition.loaders import _generate_synthetic_data

df = _generate_synthetic_data(n_consumers=500, random_state=42)

pipeline = CAPSPipeline(
    observation_window=90,
    variance_threshold=0.72,
    min_silhouette=0.45,
    random_state=42,
)

results = pipeline.fit(df)
results.summary()
results.segment_map()
```

## Using Real Data

```python
import pandas as pd
from caps import CAPSPipeline
from caps.acquisition.loaders import load_from_csv

df = load_from_csv("your_panel_data.csv")
pipeline = CAPSPipeline(observation_window=90)
results = pipeline.fit(df)
results.export("segment_assignments.csv")
```

Your CSV must contain the columns documented in `data/README.md`.

## Advanced Visualization

```python
from caps.visualization.segment_map import SegmentMapPlotter

plotter = SegmentMapPlotter(
    Z=results.reduced_matrix,
    labels=results.cluster_labels,
    profiles=results.segment_profiles,
)

plotter.plot_2d(show_ellipses=True, save_path="segment_map_2d.png")
plotter.plot_radar(feature_names=['Adoption Lag', 'Switch', 'Elasticity', 'Priority', 'Loyalty'])
plotter.plot_heatmap()

if results.n_components >= 3:
    plotter.plot_3d()
```

## Entropy Weight Inspection

```python
from caps.visualization.signal_plots import plot_entropy_weights

plot_entropy_weights(
    weights=results.entropy_weights,
    entropy=pipeline._weighter.entropy_,
    feature_names=['Adoption Lag', 'Brand Switch', 'Price Elasticity',
                   'Attr. Priority', 'Loyalty Retention'],
)
```

## Qualitative Refinement Integration

```python
from caps.refinement.qualitative import QualitativeRefinement
from caps.refinement.alignment import AlignmentValidator

qr = QualitativeRefinement(pipeline.config)

coded = qr.code_transcript(
    transcript="quality was the main thing. price did not really matter.",
    consumer_id="C00001",
    cluster_id=0,
    rater_id="R1",
)
print(coded['pp_label'])

kappa = qr.compute_kappa(
    ratings_r1=["high", "mid", "low", "high"],
    ratings_r2=["high", "mid", "mid", "high"],
)
print(f"Kappa: {kappa:.3f}")
```

## Export Results

```python
from caps.utils.io import save_results_json, save_segment_assignments

save_results_json(results, "caps_results.json")
save_segment_assignments(
    consumer_ids=results._profiler._consumer_ids,
    labels=results.cluster_labels,
    profiles=results.segment_profiles,
    path="assignments.csv",
)
```