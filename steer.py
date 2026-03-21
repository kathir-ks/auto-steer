"""
Auto-steer analysis script — THIS IS THE FILE THE AGENT MODIFIES.

Loads cached activations from prepare_steer.py and analyzes neuron firing
patterns to understand how neurons contribute to different concepts.

The agent iterates on this file: trying different analysis techniques,
probing methods, neuron importance metrics, etc.

Usage: uv run steer.py

Primary metric: interpretability_score (higher is better, range 0.0 to 1.0)
    Composite of:
    - sparsity_score:       How few neurons are needed to classify each concept (fewer = better)
    - monosemanticity_score: How cleanly neurons map to single concepts (1-to-1 = better)
    - orthogonality_score:  How independent concept directions are (orthogonal = better)
    - layer_locality_score: How concentrated concept representations are across layers
"""

import json
import time
import warnings
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from prepare_steer import (
    load_all_activations,
    get_extraction_meta,
    CONCEPTS,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results_steer")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Analysis Configuration (agent can modify these)
# ---------------------------------------------------------------------------

PROBE_CV_FOLDS = 5
PROBE_MAX_ITER = 2000

# Sparse probing: sweep over neuron budgets to find minimum needed
SPARSITY_BUDGETS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]

# Accuracy threshold to consider a concept "solved"
ACCURACY_THRESHOLD = 0.95

# Top-K neurons to report per concept
TOP_K_NEURONS = 10

# Minimum Cohen's d to consider a neuron "significant" for monosemanticity
MIN_EFFECT_SIZE = 0.5

# Composite score weights
W_SPARSITY = 0.30
W_MONOSEMANTICITY = 0.25
W_ORTHOGONALITY = 0.25
W_LAYER_LOCALITY = 0.20

# ---------------------------------------------------------------------------
# Probing Helpers
# ---------------------------------------------------------------------------

def make_dataset(pos_acts, neg_acts):
    X = np.concatenate([pos_acts, neg_acts], axis=0)
    y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
    return X, y


def probe_accuracy(X, y, C=1.0):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(C=C, max_iter=PROBE_MAX_ITER, solver="lbfgs", random_state=42)
    cv = StratifiedKFold(n_splits=PROBE_CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_s, y, cv=cv, scoring="accuracy")
    return scores.mean()


def fit_probe(X, y, C=1.0, penalty="l2"):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    solver = "saga" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(C=C, penalty=penalty, max_iter=PROBE_MAX_ITER,
                             solver=solver, random_state=42)
    clf.fit(X_s, y)
    return clf, scaler


def get_neuron_ranking(clf, scaler):
    """Return neuron indices sorted by absolute importance (descending)."""
    weights = clf.coef_[0] / scaler.scale_
    return np.argsort(np.abs(weights))[::-1], weights


# ---------------------------------------------------------------------------
# PHASE 1: Sparse Probing — minimum neurons needed per concept
# ---------------------------------------------------------------------------

def sparse_probing(all_acts, concept_names, num_layers):
    """
    For each concept, find its best layer, then determine the MINIMUM number
    of neurons needed to maintain accuracy above ACCURACY_THRESHOLD.

    Returns:
        sparse_results: dict per concept with best_layer, min_neurons, budget_curve
        sparsity_score: normalized score (0 to 1, higher = sparser = better)
    """
    print("=" * 70)
    print("PHASE 1: Sparse Probing — Minimum Neurons per Concept")
    print("=" * 70)

    sparse_results = {}
    hidden_size = None

    for concept_name in concept_names:
        # First find best layer with full probe
        best_layer = 0
        best_acc = 0.0
        for layer_idx in range(num_layers):
            pos = all_acts[concept_name]["positive"][layer_idx]
            neg = all_acts[concept_name]["negative"][layer_idx]
            X, y = make_dataset(pos, neg)
            acc = probe_accuracy(X, y)
            if acc > best_acc:
                best_acc = acc
                best_layer = layer_idx

        # At the best layer, get neuron ranking
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        hidden_size = pos.shape[1]
        X, y = make_dataset(pos, neg)
        # Use L1 probe for sparse neuron ranking — selects cleaner features
        clf, scaler = fit_probe(X, y, C=0.1, penalty="l1")
        ranked_neurons, weights = get_neuron_ranking(clf, scaler)

        # Sweep neuron budgets: use only top-k neurons
        budget_curve = {}
        min_neurons = hidden_size  # default: need all

        for budget in SPARSITY_BUDGETS:
            if budget > hidden_size:
                break
            top_k = ranked_neurons[:budget]
            X_sparse, y_sparse = make_dataset(pos[:, top_k], neg[:, top_k])
            acc_sparse = probe_accuracy(X_sparse, y_sparse, C=0.1)
            budget_curve[budget] = acc_sparse

            if acc_sparse >= ACCURACY_THRESHOLD and budget < min_neurons:
                min_neurons = budget

        sparse_results[concept_name] = {
            "best_layer": best_layer,
            "full_accuracy": best_acc,
            "min_neurons": min_neurons,
            "budget_curve": budget_curve,
            "top_neurons": [int(x) for x in ranked_neurons[:TOP_K_NEURONS]],
            "top_weights": [float(weights[x]) for x in ranked_neurons[:TOP_K_NEURONS]],
        }

        curve_str = " ".join(f"{k}:{v:.2f}" for k, v in sorted(budget_curve.items()))
        print(f"  {concept_name:20s}: layer={best_layer:2d}, "
              f"min_neurons={min_neurons:3d}, full_acc={best_acc:.3f}")
        print(f"    budget curve: {curve_str}")

    # Sparsity score: how few neurons needed, normalized
    # Score = 1 - mean(min_neurons) / hidden_size
    # A perfect score means each concept needs only 1 neuron
    mean_min = np.mean([v["min_neurons"] for v in sparse_results.values()])
    sparsity_score = 1.0 - (mean_min / hidden_size)

    print(f"\n  mean_min_neurons: {mean_min:.1f} / {hidden_size}")
    print(f"  >>> sparsity_score: {sparsity_score:.6f} <<<\n")

    return sparse_results, sparsity_score


# ---------------------------------------------------------------------------
# PHASE 2: Monosemanticity — do neurons map cleanly to single concepts?
# ---------------------------------------------------------------------------

def monosemanticity_analysis(all_acts, concept_names, sparse_results, num_layers):
    """
    For each concept's top neurons, measure how exclusively they respond to
    that concept vs all other concepts. A monosemantic neuron fires strongly
    for exactly one concept.

    Uses Cohen's d (effect size) per neuron per concept, then checks if
    each neuron's top concept is clearly dominant.

    Returns:
        mono_results: per-neuron analysis
        monosemanticity_score: 0 to 1 (higher = more monosemantic)
    """
    print("=" * 70)
    print("PHASE 2: Monosemanticity — Neuron-Concept Exclusivity")
    print("=" * 70)

    # Collect all "important" neurons across concepts (union of top-K per concept)
    important_neurons = set()
    neuron_to_concepts = {}  # neuron_idx -> {concept: effect_size}

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        for nidx in sparse_results[concept_name]["top_neurons"][:TOP_K_NEURONS]:
            important_neurons.add((best_layer, nidx))

    # For each important neuron, compute effect size for ALL concepts at that layer
    for (layer_idx, neuron_idx) in important_neurons:
        effects = {}
        for concept_name in concept_names:
            pos = all_acts[concept_name]["positive"][layer_idx][:, neuron_idx]
            neg = all_acts[concept_name]["negative"][layer_idx][:, neuron_idx]
            pooled_std = np.sqrt((pos.var() + neg.var()) / 2) + 1e-8
            cohens_d = abs(pos.mean() - neg.mean()) / pooled_std
            effects[concept_name] = cohens_d

        neuron_to_concepts[(layer_idx, neuron_idx)] = effects

    # For each neuron, compute monosemanticity using selectivity index:
    # SI = (d_max - d_mean_others) / (d_max + d_mean_others)
    # Range: -1 to 1 (1 = perfectly monosemantic, 0 = responds equally to all)
    # Then rescale to 0-1.
    mono_ratios = []
    mono_details = []

    for (layer_idx, neuron_idx), effects in neuron_to_concepts.items():
        sorted_effects = sorted(effects.items(), key=lambda x: -x[1])
        top_concept, top_d = sorted_effects[0]
        # Skip neurons with weak top effect (noise)
        if top_d < MIN_EFFECT_SIZE:
            continue
        other_ds = [d for _, d in sorted_effects[1:]]
        mean_other = np.mean(other_ds) if other_ds else 0.0
        # Selectivity index
        si = (top_d - mean_other) / (top_d + mean_other + 1e-8)
        mono_ratio = (si + 1.0) / 2.0  # rescale from [-1,1] to [0,1]
        mono_ratios.append(mono_ratio)

        mono_details.append({
            "layer": layer_idx,
            "neuron": neuron_idx,
            "top_concept": top_concept,
            "top_d": top_d,
            "mono_ratio": mono_ratio,
            "all_effects": {c: round(d, 3) for c, d in sorted_effects},
        })

    # Sort by mono_ratio descending
    mono_details.sort(key=lambda x: -x["mono_ratio"])

    # Print top monosemantic neurons
    print(f"  Analyzed {len(mono_details)} important neurons across concepts\n")
    print(f"  Top 15 most monosemantic neurons:")
    for d in mono_details[:15]:
        print(f"    L{d['layer']:02d} N{d['neuron']:3d}: "
              f"concept={d['top_concept']:20s}, d={d['top_d']:.2f}, "
              f"mono={d['mono_ratio']:.3f}")

    print(f"\n  Bottom 5 most polysemantic neurons:")
    for d in mono_details[-5:]:
        top3 = list(d["all_effects"].items())[:3]
        top3_str = ", ".join(f"{c[:8]}={v:.2f}" for c, v in top3)
        print(f"    L{d['layer']:02d} N{d['neuron']:3d}: "
              f"mono={d['mono_ratio']:.3f}, top3=[{top3_str}]")

    # Weight monosemanticity by neuron importance (top_d) — highly selective
    # important neurons should count more than weak ones
    mono_weights = np.array([d["top_d"] for d in mono_details if d["top_d"] >= MIN_EFFECT_SIZE])
    mono_vals = np.array(mono_ratios)
    if len(mono_weights) > 0:
        monosemanticity_score = float(np.average(mono_vals, weights=mono_weights))
    else:
        monosemanticity_score = float(np.mean(mono_ratios))
    print(f"\n  >>> monosemanticity_score: {monosemanticity_score:.6f} <<<\n")

    return mono_details, monosemanticity_score


# ---------------------------------------------------------------------------
# PHASE 3: Orthogonality — how independent are concept directions?
# ---------------------------------------------------------------------------

def orthogonality_analysis(all_acts, concept_names, sparse_results):
    """
    Compute steering vectors (difference of means) and measure how orthogonal
    they are. Perfect orthogonality = concepts live in completely independent
    subspaces.

    Returns:
        overlap_matrix: cosine similarity matrix
        orthogonality_score: 0 to 1 (1 = perfectly orthogonal)
    """
    print("=" * 70)
    print("PHASE 3: Concept Orthogonality")
    print("=" * 70)

    # Use logistic regression weight vectors as concept directions.
    # These are discriminatively learned and tend to be more orthogonal
    # than raw difference-of-means since the probe optimizes for classification.
    steering_vectors = {}
    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        X, y = make_dataset(pos, neg)
        clf, scaler = fit_probe(X, y, C=0.01)  # strong regularization for cleaner directions
        # The weight vector in original space
        w = clf.coef_[0] / scaler.scale_
        norm = np.linalg.norm(w) + 1e-8
        steering_vectors[concept_name] = w / norm

    n = len(concept_names)
    overlap_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vi = steering_vectors[concept_names[i]]
            vj = steering_vectors[concept_names[j]]
            overlap_matrix[i, j] = np.dot(vi, vj)

    # Print matrix
    header = "            " + "".join(f"{c[:8]:>10s}" for c in concept_names)
    print(header)
    for i, name in enumerate(concept_names):
        row = f"  {name[:10]:10s}" + "".join(f"{overlap_matrix[i,j]:10.3f}" for j in range(n))
        print(row)

    # Orthogonality score = 1 - mean |off-diagonal cosine similarity|
    off_diag = overlap_matrix[np.triu_indices(n, k=1)]
    mean_abs_overlap = np.mean(np.abs(off_diag))
    orthogonality_score = 1.0 - mean_abs_overlap

    print(f"\n  mean_abs_overlap: {mean_abs_overlap:.6f}")
    print(f"  >>> orthogonality_score: {orthogonality_score:.6f} <<<\n")

    return overlap_matrix, orthogonality_score, steering_vectors


# ---------------------------------------------------------------------------
# PHASE 4: Layer Locality — how concentrated is each concept across layers?
# ---------------------------------------------------------------------------

def layer_locality_analysis(all_acts, concept_names, num_layers):
    """
    For each concept, measure how concentrated its representation is across layers.
    A highly localized concept is detectable at only a few layers (sharp transition).
    A diffuse concept is spread across many layers.

    Uses the "sharpness" of the accuracy-vs-layer curve.

    Returns:
        locality_results: per-concept layer curves and metrics
        layer_locality_score: 0 to 1 (1 = highly concentrated)
    """
    print("=" * 70)
    print("PHASE 4: Layer Locality — Where Do Concepts Live?")
    print("=" * 70)

    locality_results = {}

    for concept_name in concept_names:
        accuracies = []
        for layer_idx in range(num_layers):
            pos = all_acts[concept_name]["positive"][layer_idx]
            neg = all_acts[concept_name]["negative"][layer_idx]
            X, y = make_dataset(pos, neg)
            acc = probe_accuracy(X, y)
            accuracies.append(acc)

        accuracies = np.array(accuracies)

        # Find the "emergence layer" — first layer where accuracy exceeds threshold
        above_threshold = np.where(accuracies >= ACCURACY_THRESHOLD)[0]
        emergence_layer = int(above_threshold[0]) if len(above_threshold) > 0 else num_layers - 1

        # Compute "concentration" using Gini coefficient on accuracy gains.
        # Idea: if accuracy jumps sharply at one layer, the gains are concentrated
        # (high Gini = high locality). If accuracy rises uniformly, Gini is low.
        gains = np.maximum(np.diff(accuracies), 0)  # per-layer accuracy gains
        gains_sorted = np.sort(gains)
        n_gains = len(gains_sorted)
        if gains_sorted.sum() < 1e-8:
            # No gains at all (accuracy is flat / already high at layer 0)
            # This means the concept is immediately available — treat as maximally local
            concentration = 1.0
        else:
            cumulative = np.cumsum(gains_sorted)
            # Gini = 1 - 2 * area under Lorenz curve
            concentration = 1.0 - 2.0 * cumulative.sum() / (n_gains * cumulative[-1])

        # Compute layer gradient — how quickly does accuracy change?
        gradients = np.abs(np.diff(accuracies))
        max_gradient = gradients.max()
        peak_gradient_layer = int(np.argmax(gradients))

        locality_results[concept_name] = {
            "emergence_layer": emergence_layer,
            "concentration": concentration,
            "max_gradient": float(max_gradient),
            "peak_gradient_layer": peak_gradient_layer,
            "layer_accuracies": [float(a) for a in accuracies],
        }

        print(f"  {concept_name:20s}: emerges@L{emergence_layer:02d}, "
              f"concentration={concentration:.3f}, "
              f"sharpest_jump@L{peak_gradient_layer:02d}({max_gradient:.3f})")

    layer_locality_score = float(np.mean([v["concentration"] for v in locality_results.values()]))
    print(f"\n  >>> layer_locality_score: {layer_locality_score:.6f} <<<\n")

    return locality_results, layer_locality_score


# ---------------------------------------------------------------------------
# PHASE 5: Neuron Role Summary — positive/negative contributions
# ---------------------------------------------------------------------------

def neuron_role_summary(all_acts, concept_names, sparse_results):
    """
    For each concept, classify neurons into positive contributors, negative
    contributors, and neutral. Summarize the firing patterns.
    """
    print("=" * 70)
    print("PHASE 5: Neuron Role Summary — Positive/Negative Contributions")
    print("=" * 70)

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        X, y = make_dataset(pos, neg)
        clf, scaler = fit_probe(X, y)
        _, weights = get_neuron_ranking(clf, scaler)

        # Classify neurons by contribution direction
        abs_w = np.abs(weights)
        threshold = np.percentile(abs_w, 95)  # top 5% are "significant"

        positive_neurons = np.where((weights > threshold))[0]
        negative_neurons = np.where((weights < -threshold))[0]
        n_significant = len(positive_neurons) + len(negative_neurons)

        # Mean activation levels for significant neurons
        pos_fire_pos = pos[:, positive_neurons].mean() if len(positive_neurons) > 0 else 0
        neg_fire_pos = neg[:, positive_neurons].mean() if len(positive_neurons) > 0 else 0
        pos_fire_neg = pos[:, negative_neurons].mean() if len(negative_neurons) > 0 else 0
        neg_fire_neg = neg[:, negative_neurons].mean() if len(negative_neurons) > 0 else 0

        top_pos = positive_neurons[np.argsort(weights[positive_neurons])[::-1][:5]] if len(positive_neurons) > 0 else []
        top_neg = negative_neurons[np.argsort(weights[negative_neurons])[:5]] if len(negative_neurons) > 0 else []

        print(f"\n  {concept_name} (layer {best_layer}):")
        print(f"    Significant neurons: {n_significant} / {len(weights)} "
              f"({len(positive_neurons)} positive, {len(negative_neurons)} negative)")
        print(f"    Positive contributors: {list(top_pos)} "
              f"(fire {pos_fire_pos:.2f} for +, {neg_fire_pos:.2f} for -)")
        print(f"    Negative contributors: {list(top_neg)} "
              f"(fire {pos_fire_neg:.2f} for +, {neg_fire_neg:.2f} for -)")


# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------

def run_analysis():
    t0 = time.time()

    meta = get_extraction_meta()
    num_layers = meta["num_layers"]
    hidden_size = meta["hidden_size"]
    concept_names = meta["concept_names"]
    print(f"Model: {meta['model_name']}, {num_layers} layers, hidden_size={hidden_size}")
    print(f"Concepts: {concept_names}")
    print(f"Prompts per direction: {meta['prompts_per_direction']}\n")

    print("Loading cached activations...")
    all_acts = load_all_activations()
    print("Loaded.\n")

    # Phase 1: Sparse probing
    sparse_results, sparsity_score = sparse_probing(all_acts, concept_names, num_layers)

    # Phase 2: Monosemanticity
    mono_details, monosemanticity_score = monosemanticity_analysis(
        all_acts, concept_names, sparse_results, num_layers
    )

    # Phase 3: Orthogonality
    overlap_matrix, orthogonality_score, steering_vectors = orthogonality_analysis(
        all_acts, concept_names, sparse_results
    )

    # Phase 4: Layer locality
    locality_results, layer_locality_score = layer_locality_analysis(
        all_acts, concept_names, num_layers
    )

    # Phase 5: Neuron role summary (informational, no score)
    neuron_role_summary(all_acts, concept_names, sparse_results)

    # ---- Composite Score ----
    interpretability_score = (
        W_SPARSITY * sparsity_score
        + W_MONOSEMANTICITY * monosemanticity_score
        + W_ORTHOGONALITY * orthogonality_score
        + W_LAYER_LOCALITY * layer_locality_score
    )

    elapsed = time.time() - t0

    # ---- Save results ----
    results = {
        "interpretability_score": float(interpretability_score),
        "sparsity_score": float(sparsity_score),
        "monosemanticity_score": float(monosemanticity_score),
        "orthogonality_score": float(orthogonality_score),
        "layer_locality_score": float(layer_locality_score),
        "weights": {"sparsity": W_SPARSITY, "monosemanticity": W_MONOSEMANTICITY,
                    "orthogonality": W_ORTHOGONALITY, "layer_locality": W_LAYER_LOCALITY},
        "elapsed_seconds": elapsed,
        "per_concept_sparse": {
            name: {
                "best_layer": v["best_layer"],
                "full_accuracy": v["full_accuracy"],
                "min_neurons": v["min_neurons"],
                "top_neurons": v["top_neurons"],
                "budget_curve": {str(k): vv for k, vv in v["budget_curve"].items()},
            }
            for name, v in sparse_results.items()
        },
        "per_concept_locality": {
            name: {
                "emergence_layer": v["emergence_layer"],
                "concentration": v["concentration"],
            }
            for name, v in locality_results.items()
        },
        "cross_concept_overlap": {
            f"{concept_names[i]}_vs_{concept_names[j]}": float(overlap_matrix[i, j])
            for i in range(len(concept_names))
            for j in range(i + 1, len(concept_names))
        },
    }

    results_path = RESULTS_DIR / "latest_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"---")
    print(f"interpretability_score: {interpretability_score:.6f}")
    print(f"sparsity_score:        {sparsity_score:.6f}")
    print(f"monosemanticity_score: {monosemanticity_score:.6f}")
    print(f"orthogonality_score:   {orthogonality_score:.6f}")
    print(f"layer_locality_score:  {layer_locality_score:.6f}")
    print(f"num_concepts:          {len(concept_names)}")
    print(f"num_layers:            {num_layers}")
    print(f"hidden_size:           {hidden_size}")
    print(f"elapsed_seconds:       {elapsed:.1f}")
    print(f"results_file:          {results_path}")
    print(f"---")

    return results


if __name__ == "__main__":
    run_analysis()
