# API Reference

## CAPSPipeline

```python
from caps import CAPSPipeline
```

The main pipeline class that orchestrates all three layers.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `observation_window` | int | 90 | Days from launch event to end of signal capture |
| `variance_threshold` | float | 0.72 | Minimum cumulative explained variance for component selection |
| `min_silhouette` | float | 0.50 | Minimum mean silhouette score for cluster acceptance |
| `min_alignment_score` | float | 0.75 | Minimum Spearman rho for qualitative-quantitative alignment |
| `min_kappa` | float | 0.70 | Minimum Cohen's kappa for inter-rater reliability |
| `n_qualitative_per_cluster` | int | 8 | Target qualitative sample per cluster |
| `qualitative_floor` | int | 8 | Minimum qualitative sample regardless of cluster size |
| `random_state` | int | 42 | Random seed for reproducibility |
| `n_jobs` | int | -1 | Parallel jobs (-1 = all cores) |
| `entropy_discretization_bins` | int | 10 | Bins for entropy computation |
| `max_clusters` | int | 12 | Maximum K for gap statistic search |
| `switching_threshold` | float | 0.30 | Expenditure reduction threshold for brand_switch |

**Methods**

`fit(df: pd.DataFrame) -> CAPSResults`

Runs the full three-layer pipeline. Returns a `CAPSResults` object.

---

## CAPSResults

Returned by `CAPSPipeline.fit()`.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `signal_matrix` | ndarray (n x p) | Raw signal matrix after acquisition |
| `weighted_matrix` | ndarray (n x p) | Entropy-weighted standardized matrix |
| `entropy_weights` | ndarray (p,) | Entropy weights per signal dimension |
| `explained_variance_ratio` | ndarray (p,) | Full EVR from PCA |
| `reduced_matrix` | ndarray (n x k) | PCA projection used for clustering |
| `n_components` | int | Number of retained components |
| `cluster_labels` | ndarray (n,) | Cluster assignment per consumer |
| `silhouette_score` | float | Mean silhouette score |
| `segment_profiles` | list[dict] | Per-segment profile dictionaries |
| `alignment_score` | float | Layer 2-3 alignment score |
| `metrics` | PipelineMetrics | Summary metrics dataclass |

**Methods**

| Method | Description |
|---|---|
| `segment_map()` | Plot 2-panel segment map |
| `summary()` | Print rich-formatted pipeline summary |
| `to_dataframe()` | Return consumer-level DataFrame with segment assignments |
| `export(path)` | Export segment assignments to CSV or Excel |

---

## SignalMatrix

```python
from caps.acquisition import SignalMatrix
```

Builds the behavioral response matrix from raw event data.

**`build(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]`**

Returns `(X, consumer_ids)`.

---

## EntropyWeighter

```python
from caps.compression import EntropyWeighter
```

Computes Shannon entropy weights for signal dimensions.

**`fit_transform(X) -> Tuple[weights, entropy]`**

**`apply(X, weights) -> X_weighted`**

**`weight_table() -> dict`**

---

## BehavioralCompressor

```python
from caps.compression import BehavioralCompressor
```

Applies PCA with variance-threshold component selection.

**`fit_transform(X_weighted) -> Tuple[Z, components, evr, n_components]`**

**`loadings_table(feature_names) -> dict`**

---

## TwoStageClustering

```python
from caps.compression import TwoStageClustering
```

Gap-statistic K selection followed by Ward-initialized K-means.

**`fit(Z) -> Tuple[labels, centroids, silhouette_score]`**

**`cluster_summary() -> dict`**

---

## QualitativeRefinement

```python
from caps.refinement import QualitativeRefinement
```

Implements the Layer 3 structured coding protocol.

**`compute_sample_size(cluster_size) -> int`**

**`select_participants(Z, labels, centroids, cluster_id) -> np.ndarray`**

**`code_transcript(transcript, consumer_id, cluster_id, rater_id) -> dict`**

**`compute_kappa(ratings_r1, ratings_r2) -> float`**

**`cluster_pp_profile(cluster_id) -> dict`**

---

## AlignmentValidator

```python
from caps.refinement import AlignmentValidator
```

Validates alignment between quantitative cluster ordering and qualitative PP profiles.

**`compute(Z, labels, components) -> float`**

**`validate_qualitative(pc1_cluster_rank, qualitative_pp_rank) -> float`**

---

## SegmentMapPlotter

```python
from caps.visualization import SegmentMapPlotter
```

Standalone visualization class with multiple plot modes.

**`plot_2d(pc_x, pc_y, show_ellipses, show_centroids)`**

**`plot_3d()`** — requires k >= 3

**`plot_radar(feature_names)`**

**`plot_heatmap(feature_names)`**