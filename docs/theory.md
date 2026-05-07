# Theoretical Basis — CAPS Framework

## The Central Inferential Claim

The CAPS Framework rests on one foundational proposition: purchasing power is
more accurately inferred from behavioral responses to competitive market pressure
than from self-reported income or demographic proxies. This is not a claim about
data availability but about measurement validity.

When a new product enters a category, it creates a forced comparison under
genuine budget constraint. Consumers cannot avoid responding — through adoption,
rejection, switching, or retention — and each response pattern carries information
about their financial and psychological latitude that no survey instrument would
reveal with equivalent fidelity.

## Layer 1 — Why Competitive Launch Events Are High-Discrimination Experiments

Standard purchasing data contains low comparative signal because consumers
operate within stable category configurations. A competitive launch disrupts this
stability, forcing consumers into a two-option comparison with real stakes. This
is the natural experimental structure that the framework exploits.

The five base signal dimensions — adoption lag, brand switching, price elasticity,
attribute priority, and loyalty retention — are not chosen arbitrarily. Each
represents a dimension along which purchasing power constrains available choice
in theoretically grounded ways. A consumer with high financial latitude faces
lower adoption cost, lower switching incentive (because they can afford the
established option), lower price sensitivity, higher quality weighting, and
higher loyalty retention through selective rather than forced switching.

## Layer 2 — Why Entropy Weighting Before PCA

Standard PCA treats all input dimensions as equally relevant to the target
construct. In the CAPS signal matrix, the five dimensions are not equally
informative about purchasing power. Price elasticity and adoption lag typically
carry higher discriminatory power than brand switching, which is affected by
category-specific loyalty norms unrelated to financial capacity.

Shannon entropy measures the information content of each dimension's distribution.
Dimensions with low entropy (concentrated distributions) carry less discriminatory
information. The entropy weight scheme penalizes these dimensions before PCA
runs, so the latent factors extracted are more tightly aligned with the
purchasing-power-relevant variance in the data.

## Layer 3 — Why Qualitative Refinement Is Structurally Necessary

A cluster of consumers sharing a position in reduced behavioral space may reflect:
- Financial homogeneity (the target construct)
- Cultural homogeneity (shared brand norms independent of capacity)
- Attitudinal homogeneity (shared category involvement levels)

These are not separable from behavioral data alone. The qualitative protocol
accesses the subjective framing of consumer decisions, which allows a trained
coder to distinguish a consumer who does not switch because they cannot afford
the new entrant from one who does not switch because they are brand committed
regardless of financial capacity.

The alignment criterion — requiring Spearman rank consistency between
quantitative cluster ordering and qualitative purchasing power ordering — is the
formal test of whether the latent factors extracted by compression correspond
to purchasing power in the real consumer population.