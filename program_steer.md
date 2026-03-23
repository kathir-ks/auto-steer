# auto-steer

Autonomous interpretability research loop. You analyze how neurons in a pretrained LLM fire and how they contribute (positively or negatively) to different concepts.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar21`). The branch `autosteer/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autosteer/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare_steer.py` — fixed utilities: model loading, concept prompts, activation extraction/caching. **Do not modify.**
   - `steer.py` — the analysis file you modify. Interpretability techniques, probing, neuron analysis.
4. **Verify cached activations exist**: Check that `~/.cache/autosteer/activations/` contains cached activation files. If not, tell the human to run `uv run prepare_steer.py`.
5. **Initialize results_steer.tsv**: Create `results_steer.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## How it works

**Model under analysis**: Qwen2.5-0.5B (~0.5B parameters, 24 layers, hidden_size=896).

**Cached activations**: `prepare_steer.py` has already run the model on ~480 contrastive prompts across 8 concept categories (sentiment, formality, certainty, temporal, complexity, subjectivity, emotion_joy_anger, instruction). Residual-stream activations at every layer (last token position) are cached as numpy arrays in `~/.cache/autosteer/activations/`.

**8 concept categories** (each with 30 positive + 30 negative prompts):
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
| `sparsity_score` | 0.30 | How few neurons are needed to classify each concept (fewer = better) | Better neuron ranking, feature selection, sparse probing |
| `monosemanticity_score` | 0.25 | How cleanly neurons map to single concepts (1:1 = better) | Activation decomposition, SAE features, better neuron identification |
| `orthogonality_score` | 0.25 | How independent concept directions are (orthogonal = better) | Concept direction estimation methods (PCA, CAVs, contrastive), denoising |
| `layer_locality_score` | 0.20 | How concentrated concept representations are across layers | Layer selection, multi-layer analysis |

All sub-scores range 0 to 1. Higher is better.

**Baseline observation**: Simple linear probes already achieve 100% classification accuracy on these concepts — the challenge is NOT accuracy, it's finding the SPARSEST, MOST MONOSEMANTIC, MOST ORTHOGONAL representations. The agent should focus on extracting cleaner, more interpretable mappings.

## Experimentation

Each experiment runs `steer.py` which operates on cached activations (no GPU/TPU needed for analysis — the heavy model inference was done once during setup). Runs take ~30-60 seconds.

**What you CAN do:**
- Modify `steer.py` — this is the only file you edit. Everything is fair game: probing methods, neuron importance metrics, analysis techniques, feature selection, concept direction estimation, etc.

**What you CANNOT do:**
- Modify `prepare_steer.py`. It is read-only.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Re-run the model inference. You work with the cached activations.

## Techniques to try

**Sparsity (finding minimal neuron sets):**
- L1-regularized probes to find truly sparse solutions
- Recursive feature elimination
- Mutual information-based feature selection
- Greedy forward/backward selection of neurons
- Group sparsity across concepts (shared sparse basis)

**Monosemanticity (1-to-1 neuron-concept maps):**
- Sparse autoencoders to decompose polysemantic neurons into monosemantic features
- Non-negative matrix factorization on activation matrices
- ICA (Independent Component Analysis) to find independent directions
- Activation thresholding to filter out weak polysemantic responses

**Orthogonality (independent concept directions):**
- PCA on concept-contrastive activations instead of difference-of-means
- Gram-Schmidt orthogonalization of steering vectors
- Concept Activation Vectors (CAVs) with explicit orthogonality constraints
- Iterative nullspace projection (INLP) for sequential concept extraction
- Cross-validated direction estimation to reduce overfitting

**Layer locality (concentrated representations):**
- Multi-layer probing with learned layer weighting
- Attention to layer transitions — where do concepts "form"?
- Residual stream decomposition (separate MLP vs attention contributions)

**Advanced:**
- Causal neuron ablation (zero out neurons, measure accuracy drop)
- Activation patching between concept pairs
- Information-theoretic measures (mutual information between neurons and concepts)
- Hierarchical clustering of neuron firing patterns
- Concept composition analysis (can you predict concept A from concept B neurons?)

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
num_layers:            24
hidden_size:           896
elapsed_seconds:       35.2
results_file:          results_steer/latest_results.json
---
```

Extract the key metric:
```
grep "^interpretability_score:" run_steer.log
```

## Logging results

When an experiment is done, log it to `results_steer.tsv` (tab-separated).

Header and columns:

```
commit	interp_score	sparsity	mono	ortho	locality	status	description
```

1. git commit hash (short, 7 chars)
2. interpretability_score (composite)
3. sparsity_score
4. monosemanticity_score
5. orthogonality_score
6. layer_locality_score
7. status: `keep`, `discard`, or `crash`
8. short text description of what this experiment tried

Example:

```
commit	interp_score	sparsity	mono	ortho	locality	status	description
a1b2c3d	0.751234	0.850000	0.650000	0.900000	0.600000	keep	baseline
b2c3d4e	0.782345	0.870000	0.700000	0.905000	0.600000	keep	L1 sparse probing with C=0.01
c3d4e5f	0.740000	0.830000	0.680000	0.895000	0.590000	discard	MLP probe (no improvement)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autosteer/mar21`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `steer.py` with an experimental idea by directly editing the code.
3. git commit
4. Run the experiment: `uv run steer.py > run_steer.log 2>&1`
5. Read out the results: `grep "^interpretability_score:\|^sparsity_score:\|^monosemanticity_score:\|^orthogonality_score:\|^layer_locality_score:" run_steer.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run_steer.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit results_steer.tsv, leave it untracked)
8. If interpretability_score improved (higher), you "advance" the branch, keeping the git commit
9. If interpretability_score is equal or worse, you git reset back to where you started

**Strategy tips:**
- The four sub-scores are somewhat independent — focus on whichever is currently lowest
- Sparsity improvements (better neuron ranking) often help monosemanticity too
- The sentiment-vs-emotion overlap (cosine ~0.52) is the biggest orthogonality drag — specifically targeting this pair will help
- Concepts emerge at very different layers (complexity@L0 vs sentiment@L10) — investigating why may yield locality insights

**Timeout**: Each analysis run should take well under 2 minutes. If a run exceeds 5 minutes, kill it and treat as failure.

**Crashes**: If a run crashes, use your judgment: fix if trivial, skip if fundamentally broken.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. You are autonomous. If you run out of ideas, think harder — look at which sub-score is weakest, try combining techniques, investigate specific concept pairs that are hard, try radically different approaches. The loop runs until the human interrupts you.
