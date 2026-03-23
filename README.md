# auto-steer

Autonomous mechanistic interpretability of concept representations in Qwen2.5-0.5B, built on top of the [autoresearch](https://github.com/karpathy/autoresearch) framework by @karpathy.

## What is this?

An AI agent (Claude) autonomously investigates how a small language model (Qwen2.5-0.5B) internally represents semantic concepts. The agent iteratively modifies `steer.py`, runs analysis, checks if the interpretability score improved, keeps or discards the change, and repeats. Over 2.5 days it ran 366 experiments and built up 2,390 analysis phases.

The approach: extract residual-stream activations from the model for 8 contrastive concept pairs (e.g., happy vs sad text, formal vs casual text), then probe those activations to understand how concepts are encoded — how many neurons are needed, how cleanly neurons map to concepts, how independent the representations are, and where in the network they emerge.

## Key findings

- **Single-neuron decoding**: Every concept is classifiable at 90%+ accuracy from just 1 neuron at the right layer
- **Perfect orthogonality**: All 8 concept directions have zero pairwise overlap — completely independent subspaces
- **Early emergence**: 5 of 8 concepts are detectable at layer 0 (the embedding layer), before any transformer processing
- **High monosemanticity**: Top neurons have selectivity scores of 0.985–0.999 (near-perfect 1-to-1 concept mapping)

See [report.md](report.md) for the full analysis with detailed results and observations.

## How it works

The repo has three files that matter:

- **`prepare_steer.py`** — one-time setup: defines 8 concepts with 30 positive/negative prompts each, runs them through Qwen2.5-0.5B, and caches residual-stream activations at all 24 layers. Not modified by the agent.
- **`steer.py`** — the analysis script the agent iterates on. Computes sparse probing, monosemanticity, orthogonality, and layer locality scores. **This file is edited and iterated on by the agent**.
- **`program_steer.md`** — instructions for the agent. **This file is edited and iterated on by the human**.

The primary metric is `interpretability_score` (0.0–1.0), a weighted composite of:
- **Sparsity** (30%): How few neurons are needed to classify each concept
- **Monosemanticity** (25%): How cleanly neurons map to single concepts
- **Orthogonality** (25%): How independent concept directions are
- **Layer locality** (20%): How concentrated representations are across layers

## The 8 concepts

| Concept | Positive | Negative |
|---------|----------|----------|
| sentiment | Happy/grateful | Sad/hopeless |
| formality | Professional register | Casual/slang |
| certainty | Confident assertions | Hedged/uncertain |
| temporal | Past-oriented | Future-oriented |
| complexity | Technical jargon | Simple/plain |
| subjectivity | Opinion/subjective | Objective/factual |
| emotion_joy_anger | Joyful/warm | Angry/hostile |
| instruction | Imperative commands | Descriptive narrative |

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). CPU is sufficient (no GPU needed for analysis).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Extract activations from Qwen2.5-0.5B (one-time, downloads model)
uv run prepare_steer.py

# 4. Run the analysis
uv run steer.py
```

## Running the agent

Spin up Claude Code (or your preferred agent) in this repo, then prompt:

```
Hi have a look at program_steer.md and let's kick off a new experiment! let's do the setup first.
```

The `program_steer.md` file provides full context and instructions for the agent.

## Project structure

```
prepare_steer.py     — activation extraction from Qwen2.5-0.5B (do not modify)
steer.py             — interpretability analysis (agent modifies this)
program_steer.md     — agent instructions
report.md            — full experiment report with findings
results_steer.tsv    — experiment log (366 entries with scores)
results_steer/       — detailed results JSON
run_steer.log        — output from latest full run
pyproject.toml       — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `steer.py`. This keeps the scope manageable and diffs reviewable.
- **Composite metric.** Four sub-scores (sparsity, monosemanticity, orthogonality, locality) provide a balanced optimization target that resists Goodhart's law better than a single metric.
- **Contrastive probing.** Each concept is defined by opposing directions (positive/negative prompts), enabling linear probing and steering vector extraction.
- **Self-contained.** No GPU needed for analysis — activations are cached once, then all analysis runs on CPU with numpy/sklearn.

## Based on

This project adapts the [autoresearch](https://github.com/karpathy/autoresearch) framework from autonomous LLM training optimization to autonomous mechanistic interpretability research.

## License

MIT
