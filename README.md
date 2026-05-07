# CAPS Framework

**Competitive-signal Anchored Purchasing-power Segmentation**

A Python library implementing the CAPS Framework for inferring purchasing power
patterns from competitive market signals. The framework integrates behavioral
dimensionality reduction with structured qualitative refinement to produce
segment profiles with ecological validity that conventional measurement approaches
cannot match.

## Background

CAPS is built on the theoretical proposition that competitive product launch events
constitute high-discrimination natural experiments. When a new product enters an
established category, the resulting behavioral signals — adoption speed, switching
patterns, price elasticity responses, attribute weighting, and loyalty retention —
form a multi-dimensional matrix from which purchasing power strata can be
systematically inferred without relying on self-reported income or demographic
proxies.

The framework operates through three sequential layers:

1. **Signal Acquisition Layer** — Structures raw behavioral responses to competitive
   launch events into a matrix X of dimension n x p.
2. **Behavioral Compression Layer** — Applies entropy-weighted PCA and two-stage
   clustering to extract purchasing-power-relevant latent structure.
3. **Qualitative Refinement Layer** — Validates and semantically labels cluster
   assignments through structured depth-interview coding and alignment scoring.

## Installation

```bash
pip install caps-framework
```

From source:

```bash
git clone https://github.com/ahmed-elsayed-99/caps-framework.git
cd caps-framework
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from caps import CAPSPipeline

df = pd.read_csv("data/raw/sample_launch_event.csv")

pipeline = CAPSPipeline(
    observation_window=90,
    variance_threshold=0.72,
    min_silhouette=0.50,
    n_qualitative_per_cluster=8
)

results = pipeline.fit(df)
results.segment_map()
results.summary()
```

## Real Data

The library ships with a synthetic dataset modelled on publicly available
Nielsen-format retail panel structures. Real applications require longitudinal
consumer-level data across a minimum observation window of 60 days spanning
a competitive product launch event. See `data/README.md` for schema documentation.

## Citation

```bibtex
@article{elsayed2026caps,
  title={Inferring Purchasing Power Patterns from Competitive Market Signals:
         A Conceptual Framework Integrating Behavioral Dimensionality Reduction
         and Qualitative Refinement},
  author={Elsayed, Ahmed},
  journal={SSRN},
  year={2026},
  note={CAPS Framework v1.0.0},
  url={https://github.com/ahmed-elsayed-99/caps-framework}
}
```

## License

MIT License. See `LICENSE`.
