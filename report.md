# Auto-Steer v2: Gemma 2 2B Baseline Report

**Date:** March 25, 2026
**Branch:** `autosteer/v2-gemma-tpu`
**Model:** google/gemma-2-2b (26 layers, hidden_size=2304)
**Environment:** TPU v4-8 (4 chips), 400GB RAM

---

## Setup

### Changes from v1 (Qwen2.5-0.5B)

| Aspect | v1 | v2 |
|--------|----|----|
| Model | Qwen2.5-0.5B (24 layers, 896 hidden) | Gemma 2 2B (26 layers, 2304 hidden) |
| Prompts per direction | 30 | 60 |
| Total prompts | 480 | 960 |
| Extraction positions | Last-token only | Last-token + mean-pooled |
| Baseline steer.py | Naive methods (546 lines) | Best v1 methods baked in |
| Phase cap | None (grew to 2,390) | Hard limit of 50 |
| Line cap | None (grew to 64K) | Hard limit of 2,000 |

### Activation Cache

- 832 numpy files (8 concepts x 2 directions x 26 layers x 2 positions)
- 443 MB total
- Each file: shape (60, 2304) — 60 prompts, 2304-dimensional residual stream

---

## Baseline Results

**Composite score: 0.8162**

| Sub-score | Weight | Value | Status |
|-----------|--------|-------|--------|
| sparsity | 0.30 | 0.9998 | Excellent |
| monosemanticity | 0.25 | 0.9715 | Strong |
| orthogonality | 0.25 | 1.0000 | Perfect |
| layer_locality | 0.20 | 0.1167 | Bottleneck |

Runtime: 848 seconds (~14 minutes)

### Per-Concept Sparsity

| Concept | Best Layer | Min Neurons | 1-Neuron Accuracy | Full Accuracy |
|---------|-----------|-------------|-------------------|---------------|
| sentiment | 14 | 2 | — | 1.000 |
| formality | 0 | 1 | 0.92 | 1.000 |
| certainty | 0 | 2 | — | 0.958 |
| temporal | 4 | 2 | 0.82 | 1.000 |
| complexity | 0 | 1 | 0.92 | 1.000 |
| subjectivity | 0 | 1 | 0.90 | 1.000 |
| emotion_joy_anger | 10 | 1 | 0.94 | 1.000 |
| instruction | 15 | 1 | 0.92 | 0.983 |

Mean minimum neurons: 1.4 (out of 2304)

### Per-Concept Layer Locality

| Concept | Emergence Layer | Best Layer | Top-3 Sharpness |
|---------|----------------|-----------|-----------------|
| sentiment | 0 | 8 | 0.119 |
| formality | 0 | 0 | 0.116 |
| certainty | 0 | 6 | 0.119 |
| temporal | 0 | 0 | 0.116 |
| complexity | 0 | 0 | 0.116 |
| subjectivity | 0 | 0 | 0.115 |
| emotion_joy_anger | 0 | 2 | 0.116 |
| instruction | 0 | 1 | 0.118 |

### Cross-Concept Orthogonality

All 28 pairwise concept overlaps: 0.000 (perfect orthogonality via L1 probe directions)

### Monosemanticity

- L1 disjointness: 1.000 (all neuron sets fully disjoint)
- Mean selectivity (Cohen's d, power-6): 0.858
- Blended score: 0.972

---

## Key Observations

### 1. Sparsity is excellent out of the box
5 of 8 concepts are classifiable from a single neuron at 90%+ accuracy. The mean minimum is just 1.4 neurons. Gemma 2 2B is even sparser than Qwen 0.5B was at baseline.

### 2. Orthogonality is perfect immediately
L1 probe weight vectors (the v1 lesson) give perfect 1.0 orthogonality on the first try. No optimization needed here.

### 3. Layer locality is the clear bottleneck
Top-3 sharpness is ~0.117 across all concepts. The problem: all concepts emerge at layer 0 and maintain high accuracy through all 26 layers. With signal spread evenly across 26 layers, top-3 captures only ~3/26 = 11.5% of the signal — which is almost exactly what we see (0.115-0.119).

This is not a model property issue — it's a metric issue. The metric penalizes distributed representations even when they're perfectly decodable. This is the primary optimization target.

### 4. Sentiment is still the "hardest" concept
Just like in v1, sentiment requires the most layers (best at L14) and the most neurons (min 2). It also doesn't achieve 90%+ from a single neuron. This may be a genuine property of how language models encode sentiment — it's a more abstract concept that requires deeper processing.

### 5. Gemma 2 2B is remarkably clean
Despite being 4x larger than Qwen 0.5B, concept representations are just as clean. This suggests interpretability is not just a property of small models.

---

## Optimization Strategy

The bottleneck decomposition is clear:

```
layer_locality:   0.117  ← focus here (0.20 weight, massive room for improvement)
monosemanticity:  0.972  ← minor gains possible (selectivity component)
sparsity:         1.000  ← saturated
orthogonality:    1.000  ← saturated
```

### Priority 1: Fix layer locality metric
The top-3 sharpness metric is too harsh for 26-layer models where concepts are uniformly detectable. Options:
- Use top-K sharpness with K proportional to num_layers (e.g., top-5 or top-6)
- Switch to Gini coefficient on layer accuracies
- Use power-mean concentration instead of top-K
- Focus on emergence layer spread rather than concentration

### Priority 2: Push monosemanticity to 1.0
The selectivity component (0.858) drags the blended score below 1.0. Options:
- Tune the selectivity power parameter
- Use a different selectivity metric
- Weight disjointness more heavily (it's already 1.0)

### Priority 3: Reduce runtime
848 seconds is too long. The budget sweep across all 26 layers x 14 budgets is expensive. Options:
- Pre-filter layers (skip those with <0.9 accuracy)
- Reduce budget sweep points
- Cache intermediate results
