# auto-steer v2

Autonomous mechanistic interpretability of concept representations in **Gemma 2 2B**, built on top of the [autoresearch](https://github.com/karpathy/autoresearch) framework by @karpathy.

## What is this?

An AI agent (Claude) autonomously investigates how a language model (Gemma 2 2B) internally represents semantic concepts. The agent iteratively modifies `steer.py`, runs analysis, checks if the interpretability score improved, keeps or discards the change, and repeats.

The approach: extract residual-stream activations from the model for 8 contrastive concept pairs (e.g., happy vs sad text, formal vs casual text), then probe those activations to understand how concepts are encoded — how many neurons are needed, how cleanly neurons map to concepts, how independent the representations are, and where in the network they emerge.

## v2 changes from v1

- **Larger model**: Gemma 2 2B (26 layers, hidden_size=2304) instead of Qwen2.5-0.5B (24 layers, hidden_size=896)
- **More data**: 60 prompts per direction (up from 30) — 960 total prompts
- **Dual extraction**: Both last-token and mean-pooled activations cached
- **Better baseline**: steer.py starts with the best methods discovered in v1 (L1 probes, sharpness locality, disjointness mono)
- **Anti-redundancy guardrails**: Hard cap of 50 phases / 2000 lines to prevent the bloat problem from v1 (which grew to 2,390 phases / 64K lines with diminishing returns)
- **TPU environment**: Running on v4-8 TPUs (4 chips)

## How it works

The repo has three files that matter:

- **`prepare_steer.py`** — one-time setup: defines 8 concepts with 60 positive/negative prompts each, runs them through Gemma 2 2B, and caches residual-stream activations at all 26 layers (both last-token and mean-pooled). Not modified by the agent.
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

**Requirements:** Python 3.10+, HuggingFace account with Gemma access. Running on TPU v4-8 but CPU works for analysis.

```bash
# 1. Install dependencies
pip3 install torch transformers scikit-learn sentencepiece flax

# 2. Login to HuggingFace (needed for gated Gemma model)
python3 -c "from huggingface_hub import login; login()"

# 3. Extract activations from Gemma 2 2B (one-time, downloads model)
python3 prepare_steer.py

# 4. Run the analysis
python3 steer.py
```

## Running the agent

Spin up Claude Code (or your preferred agent) in this repo, then prompt:

```
Hi have a look at program_steer.md and let's kick off a new experiment! let's do the setup first.
```

## Project structure

```
prepare_steer.py     — activation extraction from Gemma 2 2B (do not modify)
steer.py             — interpretability analysis (agent modifies this)
program_steer.md     — agent instructions (with anti-redundancy rules)
pyproject.toml       — dependencies
```

## Based on

This project adapts the [autoresearch](https://github.com/karpathy/autoresearch) framework from autonomous LLM training optimization to autonomous mechanistic interpretability research. See the `autosteer/mar21` branch for the v1 experiment on Qwen2.5-0.5B.

## License

MIT
