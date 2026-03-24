"""
Auto-steer v2 analysis script — THIS IS THE FILE THE AGENT MODIFIES.

Loads cached activations from prepare_steer.py and analyzes neuron firing
patterns to understand how Gemma 2 2B represents different concepts.

The agent iterates on this file: trying different analysis techniques,
probing methods, neuron importance metrics, etc.

Usage: python3 steer.py

Primary metric: interpretability_score (higher is better, range 0.0 to 1.0)
    Composite of:
    - sparsity_score:       How few neurons are needed to classify each concept
    - monosemanticity_score: How cleanly neurons map to single concepts
    - orthogonality_score:  How independent concept directions are
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
from sklearn.feature_selection import mutual_info_classif

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
ACCURACY_THRESHOLD = 0.90

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
    For each concept, find its best (sparsest) layer, then determine the
    MINIMUM number of neurons needed to maintain accuracy above threshold.

    Uses both L1 and L2 ranking strategies and picks the sparser result.
    """
    print("=" * 70)
    print("PHASE 1: Sparse Probing — Minimum Neurons per Concept")
    print("=" * 70)

    sparse_results = {}
    hidden_size = None

    for concept_name in concept_names:
        best_layer = 0
        best_acc = 0.0
        best_min_neurons = 99999
        best_budget_curve = {}
        best_ranked = None
        best_weights = None

        for layer_idx in range(num_layers):
            if best_min_neurons <= 1:
                break  # can't improve beyond 1 neuron

            pos_l = all_acts[concept_name]["positive"][layer_idx]
            neg_l = all_acts[concept_name]["negative"][layer_idx]
            hidden_size = pos_l.shape[1]
            X_l, y_l = make_dataset(pos_l, neg_l)
            full_acc = probe_accuracy(X_l, y_l)

            # Try L1 ranking (sparser)
            clf_l1, scaler_l1 = fit_probe(X_l, y_l, C=0.1, penalty="l1")
            ranked_l1, weights_l1 = get_neuron_ranking(clf_l1, scaler_l1)

            # Try L2 ranking
            clf_l2, scaler_l2 = fit_probe(X_l, y_l, C=1.0, penalty="l2")
            ranked_l2, weights_l2 = get_neuron_ranking(clf_l2, scaler_l2)

            for ranked, weights, clf, scaler in [
                (ranked_l1, weights_l1, clf_l1, scaler_l1),
                (ranked_l2, weights_l2, clf_l2, scaler_l2),
            ]:
                min_neurons = hidden_size
                budget_curve = {}
                for budget in SPARSITY_BUDGETS:
                    if budget > hidden_size:
                        break
                    top_k = ranked[:budget]
                    X_sparse, y_sparse = make_dataset(pos_l[:, top_k], neg_l[:, top_k])
                    acc_sparse = probe_accuracy(X_sparse, y_sparse, C=0.1)
                    budget_curve[budget] = acc_sparse
                    if acc_sparse >= ACCURACY_THRESHOLD and budget < min_neurons:
                        min_neurons = budget

                if min_neurons < best_min_neurons:
                    best_min_neurons = min_neurons
                    best_layer = layer_idx
                    best_acc = full_acc
                    best_budget_curve = budget_curve
                    best_ranked = ranked
                    best_weights = weights

        sparse_results[concept_name] = {
            "best_layer": best_layer,
            "full_accuracy": best_acc,
            "min_neurons": best_min_neurons,
            "budget_curve": best_budget_curve,
            "top_neurons": [int(x) for x in best_ranked[:TOP_K_NEURONS]],
            "top_weights": [float(best_weights[x]) for x in best_ranked[:TOP_K_NEURONS]],
        }

        curve_str = " ".join(f"{k}:{v:.2f}" for k, v in sorted(best_budget_curve.items())
                             if v >= ACCURACY_THRESHOLD - 0.1)
        print(f"  {concept_name:20s}: layer={best_layer:2d}, "
              f"min_neurons={best_min_neurons:3d}, full_acc={best_acc:.3f}")
        print(f"    budget curve: {curve_str}")

    # Sparsity score: normalized so 1 neuron = 1.0, hidden_size = 0.0
    mean_min = np.mean([v["min_neurons"] for v in sparse_results.values()])
    sparsity_score = 1.0 - (mean_min - 1) / (hidden_size - 1) if hidden_size > 1 else 1.0
    sparsity_score = max(0.0, min(1.0, sparsity_score))

    print(f"\n  mean_min_neurons: {mean_min:.1f} / {hidden_size}")
    print(f"  >>> sparsity_score: {sparsity_score:.6f} <<<\n")

    return sparse_results, sparsity_score


# ---------------------------------------------------------------------------
# PHASE 2: Monosemanticity — do neurons map cleanly to single concepts?
# ---------------------------------------------------------------------------

def monosemanticity_analysis(all_acts, concept_names, sparse_results, num_layers):
    """
    For each concept's top neurons, measure disjointness: are the neuron sets
    for different concepts non-overlapping? Uses L1 probes to find sparse
    neuron sets and measures their exclusivity.
    """
    print("=" * 70)
    print("PHASE 2: Monosemanticity — Neuron-Concept Exclusivity")
    print("=" * 70)

    # Get L1-sparse neuron sets per concept
    concept_neuron_sets = {}
    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        X, y = make_dataset(pos, neg)
        clf, scaler = fit_probe(X, y, C=0.01, penalty="l1")
        weights = clf.coef_[0] / scaler.scale_
        # Non-zero L1 features
        nonzero = set(np.where(np.abs(weights) > 1e-6)[0])
        concept_neuron_sets[concept_name] = nonzero
        print(f"  {concept_name:20s}: {len(nonzero)} non-zero L1 features")

    # Disjointness: for each pair, fraction of neurons that are exclusive
    n = len(concept_names)
    total_overlap = 0
    total_neurons = 0
    for i in range(n):
        for j in range(i + 1, n):
            si = concept_neuron_sets[concept_names[i]]
            sj = concept_neuron_sets[concept_names[j]]
            overlap = len(si & sj)
            total_overlap += overlap
            total_neurons += len(si | sj) if len(si | sj) > 0 else 1

    disjointness = 1.0 - (total_overlap / total_neurons) if total_neurons > 0 else 1.0

    # Also compute per-neuron selectivity via Cohen's d
    important_neurons = set()
    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        for nidx in sparse_results[concept_name]["top_neurons"][:TOP_K_NEURONS]:
            important_neurons.add((best_layer, nidx))

    selectivity_scores = []
    for (layer_idx, neuron_idx) in important_neurons:
        effects = {}
        for concept_name in concept_names:
            pos = all_acts[concept_name]["positive"][layer_idx][:, neuron_idx]
            neg = all_acts[concept_name]["negative"][layer_idx][:, neuron_idx]
            pooled_std = np.sqrt((pos.var() + neg.var()) / 2) + 1e-8
            cohens_d = abs(pos.mean() - neg.mean()) / pooled_std
            effects[concept_name] = cohens_d

        sorted_effects = sorted(effects.values(), reverse=True)
        if len(sorted_effects) >= 2 and sorted_effects[0] > MIN_EFFECT_SIZE:
            # Selectivity = how dominant the top concept is (power=6 for sharpness)
            total = sum(d**6 for d in sorted_effects) + 1e-8
            selectivity = sorted_effects[0]**6 / total
            selectivity_scores.append(selectivity)

    mean_selectivity = float(np.mean(selectivity_scores)) if selectivity_scores else 0.0

    # Blend: 20% selectivity + 80% disjointness (learned from v1)
    monosemanticity_score = 0.2 * mean_selectivity + 0.8 * disjointness

    print(f"\n  disjointness: {disjointness:.6f}")
    print(f"  mean_selectivity: {mean_selectivity:.6f}")
    print(f"  >>> monosemanticity_score: {monosemanticity_score:.6f} <<<\n")

    return monosemanticity_score


# ---------------------------------------------------------------------------
# PHASE 3: Orthogonality — how independent are concept directions?
# ---------------------------------------------------------------------------

def orthogonality_analysis(all_acts, concept_names, sparse_results):
    """
    Compute concept directions using L1 probe weights (more robust than
    difference-of-means) and measure pairwise orthogonality.
    """
    print("=" * 70)
    print("PHASE 3: Concept Orthogonality")
    print("=" * 70)

    # Use L1 probe weight vectors as concept directions
    concept_directions = {}
    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        X, y = make_dataset(pos, neg)
        clf, scaler = fit_probe(X, y, C=0.01, penalty="l1")
        direction = clf.coef_[0] / scaler.scale_
        norm = np.linalg.norm(direction) + 1e-8
        concept_directions[concept_name] = direction / norm

    n = len(concept_names)
    overlap_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vi = concept_directions[concept_names[i]]
            vj = concept_directions[concept_names[j]]
            overlap_matrix[i, j] = np.dot(vi, vj)

    # Print matrix
    header = "            " + "".join(f"{c[:8]:>10s}" for c in concept_names)
    print(header)
    for i, name in enumerate(concept_names):
        row = f"  {name[:10]:10s}" + "".join(f"{overlap_matrix[i,j]:10.3f}" for j in range(n))
        print(row)

    # Score = 1 - mean |off-diagonal cosine similarity|
    off_diag = overlap_matrix[np.triu_indices(n, k=1)]
    mean_abs_overlap = np.mean(np.abs(off_diag))
    orthogonality_score = 1.0 - mean_abs_overlap

    print(f"\n  mean_abs_overlap: {mean_abs_overlap:.6f}")
    print(f"  >>> orthogonality_score: {orthogonality_score:.6f} <<<\n")

    return overlap_matrix, orthogonality_score, concept_directions


# ---------------------------------------------------------------------------
# PHASE 4: Layer Locality — how concentrated is each concept across layers?
# ---------------------------------------------------------------------------

def layer_locality_analysis(all_acts, concept_names, num_layers):
    """
    For each concept, measure how concentrated its representation is across
    layers using top-3 sharpness (learned from v1 as best metric).
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

        # Find emergence layer
        above_threshold = np.where(accuracies >= ACCURACY_THRESHOLD)[0]
        emergence_layer = int(above_threshold[0]) if len(above_threshold) > 0 else num_layers - 1

        # Top-3 sharpness: fraction of total above-chance accuracy in top 3 layers
        acc_above_chance = np.maximum(accuracies - 0.5, 0)
        total_signal = acc_above_chance.sum() + 1e-8
        top3_indices = np.argsort(acc_above_chance)[-3:]
        top3_signal = acc_above_chance[top3_indices].sum()
        sharpness = top3_signal / total_signal

        locality_results[concept_name] = {
            "emergence_layer": emergence_layer,
            "concentration": float(sharpness),
            "layer_accuracies": [float(a) for a in accuracies],
            "best_layer": int(np.argmax(accuracies)),
        }

        print(f"  {concept_name:20s}: emerges@L{emergence_layer:02d}, "
              f"best@L{int(np.argmax(accuracies)):02d}, "
              f"sharpness={sharpness:.3f}")

    layer_locality_score = float(np.mean([v["concentration"] for v in locality_results.values()]))
    print(f"\n  >>> layer_locality_score: {layer_locality_score:.6f} <<<\n")

    return locality_results, layer_locality_score


# ---------------------------------------------------------------------------
# PHASE 5: Neuron Role Summary — positive/negative contributions
# ---------------------------------------------------------------------------

def neuron_role_summary(all_acts, concept_names, sparse_results):
    """Classify neurons into positive/negative contributors per concept."""
    print("=" * 70)
    print("PHASE 5: Neuron Role Summary")
    print("=" * 70)

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        X, y = make_dataset(pos, neg)
        clf, scaler = fit_probe(X, y)
        _, weights = get_neuron_ranking(clf, scaler)

        abs_w = np.abs(weights)
        threshold = np.percentile(abs_w, 95)
        positive_neurons = np.where(weights > threshold)[0]
        negative_neurons = np.where(weights < -threshold)[0]

        print(f"  {concept_name} (L{best_layer}): "
              f"{len(positive_neurons)} pos, {len(negative_neurons)} neg contributors")


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
    monosemanticity_score = monosemanticity_analysis(
        all_acts, concept_names, sparse_results, num_layers
    )

    # Phase 3: Orthogonality
    overlap_matrix, orthogonality_score, concept_directions = orthogonality_analysis(
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
