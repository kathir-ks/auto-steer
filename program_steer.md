# auto-steer v2

Autonomous interpretability research loop. You analyze how neurons in Gemma 2 2B fire and how they contribute to different concepts.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar24`). The branch `autosteer/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autosteer/<tag>` from current branch.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare_steer.py` — fixed utilities: model loading, concept prompts, activation extraction/caching. **Do not modify.**
   - `steer.py` — the analysis file you modify. Interpretability techniques, probing, neuron analysis.
4. **Verify cached activations exist**: Check that `~/.cache/autosteer-v2/activations/` contains cached activation files. If not, tell the human to run `python3 prepare_steer.py`.
5. **Initialize results_steer.tsv**: Create `results_steer.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## How it works

**Model under analysis**: Gemma 2 2B (~2B parameters, 26 layers, hidden_size=2304).

**Cached activations**: `prepare_steer.py` has already run the model on ~960 contrastive prompts across 8 concept categories. Residual-stream activations at every layer are cached as numpy arrays in `~/.cache/autosteer-v2/activations/`. Both **last-token** and **mean-pooled** positions are available.

**8 concept categories** (each with 60 positive + 60 negative prompts):
- **sentiment**: positive vs negative emotional tone
- **formality**: formal professional vs casual informal
- **certainty**: confident assertions vs uncertain hedging
- **temporal**: past-oriented vs future-oriented language
- **complexity**: technical/complex vs simple/plain language
- **subjectivity**: subjective opinion vs objective fact
- **emotion_joy_anger**: joyful/warm vs angry/hostile emotion
- **instruction**: imperative instructions vs descriptive narratives

**What you analyze**: How neurons at each layer fire differently for these concepts. Which neurons contribute positively or negatively to each concept. Whether concept directions are cleanly separated or entangled. How sparse and monosemantic the neuron-concept mappings are.

## The metric

The primary metric is `interpretability_score` — a composite of four sub-scores:

| Sub-score | Weight | What it measures | How to improve it |
|---|---|---|---|
| `sparsity_score` | 0.30 | How few neurons needed per concept (fewer = better) | Better neuron ranking, L1 probes, MI-based selection |
| `monosemanticity_score` | 0.25 | How cleanly neurons map to single concepts (1:1 = better) | Disjointness of neuron sets, selectivity metrics |
| `orthogonality_score` | 0.25 | How independent concept directions are (orthogonal = better) | L1 probe directions, whitening, Gram-Schmidt |
| `layer_locality_score` | 0.20 | How concentrated representations are across layers | Top-K sharpness, layer selection |

All sub-scores range 0 to 1. Higher is better.

## CRITICAL: Anti-redundancy rules

**Lessons from v1**: The previous experiment hit a perfect 1.0 score after 65 iterations, then spent 300+ more iterations adding 2,300 redundant analysis phases that produced no new insight. The file bloated from 546 lines to 64,374 lines. DO NOT repeat this pattern.

**Hard rules:**

1. **Phase cap**: `steer.py` must not exceed **50 analysis phases** and **2000 lines of code**. If you hit these limits, STOP adding phases and focus purely on improving existing ones.

2. **No redundant phases**: Before adding a new analysis phase, check if an existing phase already computes the same or very similar metric. If so, modify the existing phase instead of adding a new one.

3. **Every phase must either improve the score or provide actionable insight**. A phase that just prints a number without informing your next experiment is wasted work. Remove it.

4. **Score plateau = pivot, not pad**: If the score stops improving for 5 consecutive experiments, do NOT add more analysis phases. Instead:
   - Try a radically different approach to the weakest sub-score
   - Try different probing methods (nonlinear, ensemble)
   - Investigate specific concept pairs that are hard
   - Try mean-pooled activations instead of last-token
   - If truly stuck, report findings and stop

5. **Quality over quantity**: One well-designed experiment that moves the score is worth more than 50 that don't.

## Experimentation

Each experiment runs `steer.py` which operates on cached activations (no GPU/TPU needed for analysis). Runs should take under 2 minutes.

**What you CAN do:**
- Modify `steer.py` — this is the only file you edit. Everything is fair game: probing methods, neuron importance metrics, analysis techniques, feature selection, concept direction estimation, etc.
- Switch between `last` and `mean` activation positions: `load_all_activations(position="mean")`

**What you CANNOT do:**
- Modify `prepare_steer.py`. It is read-only.
- Add new pip dependencies beyond what's already installed.
- Re-run the model inference. You work with cached activations.

## Techniques to try

**Sparsity:**
- L1-regularized probes (C=0.01 to C=10 sweep)
- Mutual information-based neuron ranking
- Greedy forward/backward neuron selection
- Budget sweep across multiple layers to find sparsest

**Monosemanticity:**
- L1 probe disjointness (non-overlapping neuron sets per concept)
- Cohen's d selectivity with power scaling
- ICA / NMF decomposition of activation matrices

**Orthogonality:**
- L1 probe weight vectors as concept directions
- Whitened cosine similarity
- Iterative nullspace projection (INLP)

**Layer locality:**
- Top-K layer sharpness
- Power-mean aggregation
- Gini coefficient on layer accuracy distribution

## Output format

The script prints a summary at the end:

```
---
interpretability_score: 0.750000
sparsity_score:        0.850000
monosemanticity_score: 0.650000
orthogonality_score:   0.900000
layer_locality_score:  0.600000
num_concepts:          8
num_layers:            26
hidden_size:           2304
elapsed_seconds:       45.2
results_file:          results_steer/latest_results.json
---
```

Extract the key metric:
```
grep "^interpretability_score:" run_steer.log
```

## Logging results

Log experiments to `results_steer.tsv` (tab-separated, untracked).

```
commit	interp_score	sparsity	mono	ortho	locality	status	description
a1b2c3d	0.751	0.850	0.650	0.900	0.600	keep	baseline
```

## The experiment loop

LOOP:

1. Look at the current state and identify which sub-score to target
2. Modify `steer.py` with an experimental idea
3. git commit
4. Run: `python3 steer.py > run_steer.log 2>&1`
5. Read results: `grep "^interpretability_score:\|^sparsity_score:\|^monosemanticity_score:\|^orthogonality_score:\|^layer_locality_score:" run_steer.log`
6. If grep is empty, run crashed — `tail -n 50 run_steer.log` to diagnose
7. Record in results_steer.tsv
8. If score improved: keep the commit
9. If score equal or worse: git reset back

**Strategy:**
- Focus on whichever sub-score is currently lowest
- Gemma 2 2B is bigger than Qwen 0.5B (2304 hidden vs 896) — may need more aggressive sparsity
- 26 layers (vs 24) and larger hidden size means richer representations but harder to find single-neuron solutions
- Try mean-pooled activations if last-token results plateau

**Timeout**: Each run should take < 2 minutes. Kill and treat as failure if > 5 minutes.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human. You are autonomous. If stuck, pivot strategy rather than padding with redundant phases.
