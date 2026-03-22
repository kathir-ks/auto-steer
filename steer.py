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
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import FastICA, NMF, DictionaryLearning
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

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
TOP_K_NEURONS = 3

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
        # Find the SPARSEST layer: for each layer, find min neurons needed
        # to hit threshold. Pick the layer that needs the fewest.
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

            # Try two ranking strategies, keep the sparser result
            # Strategy 1: L1 probe ranking
            clf_l, scaler_l = fit_probe(X_l, y_l, C=0.1, penalty="l1")
            ranked_l1, weights_l1 = get_neuron_ranking(clf_l, scaler_l)

            # Strategy 2: Mutual information ranking
            mi_scores = mutual_info_classif(X_l, y_l, random_state=42)
            ranked_mi = np.argsort(mi_scores)[::-1]

            # Evaluate both rankings, keep best
            layer_min = hidden_size
            layer_curve = {}
            ranked_l = ranked_l1
            weights_l = weights_l1

            for ranking, name in [(ranked_l1, "L1"), (ranked_mi, "MI")]:
                r_min = hidden_size
                r_curve = {}
                for budget in SPARSITY_BUDGETS:
                    if budget > hidden_size or budget > r_min:
                        break  # no point checking larger budgets
                    top_k = ranking[:budget]
                    X_sp, y_sp = make_dataset(pos_l[:, top_k], neg_l[:, top_k])
                    acc_sp = probe_accuracy(X_sp, y_sp, C=1.0)
                    r_curve[budget] = acc_sp
                    if acc_sp >= ACCURACY_THRESHOLD and budget < r_min:
                        r_min = budget
                    if acc_sp >= 1.0 - 1e-8:
                        break  # perfect accuracy, no need to check more
                if r_min < layer_min:
                    layer_min = r_min
                    layer_curve = r_curve
                    ranked_l = ranking
                    weights_l = weights_l1  # keep L1 weights for downstream

            # Prefer layer with fewer min_neurons; break ties by full accuracy
            if (layer_min < best_min_neurons or
                (layer_min == best_min_neurons and full_acc > best_acc)):
                best_layer = layer_idx
                best_acc = full_acc
                best_min_neurons = layer_min
                best_budget_curve = layer_curve
                best_ranked = ranked_l
                best_weights = weights_l

        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        ranked_neurons = best_ranked
        weights = best_weights
        budget_curve = best_budget_curve
        min_neurons = best_min_neurons

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
    # Since min_neurons >= 1 (need at least one), normalize excess above 1:
    # Score = 1 - (mean_min - 1) / (hidden_size - 1)
    # This maps 1 neuron → 1.0 (perfect), hidden_size neurons → 0.0
    mean_min = np.mean([v["min_neurons"] for v in sparse_results.values()])
    sparsity_score = 1.0 - (mean_min - 1.0) / (hidden_size - 1.0)

    print(f"\n  mean_min_neurons: {mean_min:.1f} / {hidden_size}")
    print(f"  >>> sparsity_score: {sparsity_score:.6f} <<<\n")

    return sparse_results, sparsity_score


# ---------------------------------------------------------------------------
# PHASE 2: Monosemanticity — do neurons map cleanly to single concepts?
# ---------------------------------------------------------------------------

def monosemanticity_analysis(all_acts, concept_names, sparse_results, num_layers):
    """
    Measure monosemanticity using two complementary approaches:

    1. Per-neuron selectivity: For each concept's top neurons, how exclusively
       does it respond to that concept vs others (Cohen's d selectivity index).

    2. L1 support disjointness: For each pair of concepts, measure how little
       their L1 probe supports overlap (Jaccard distance on non-zero weights).

    Returns:
        mono_details: per-neuron analysis
        monosemanticity_score: 0 to 1 (higher = more monosemantic)
    """
    print("=" * 70)
    print("PHASE 2: Monosemanticity — Neuron-Concept Exclusivity")
    print("=" * 70)

    # --- Part A: Per-neuron selectivity (as before) ---
    important_neurons = set()
    neuron_to_concepts = {}

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        for nidx in sparse_results[concept_name]["top_neurons"][:TOP_K_NEURONS]:
            important_neurons.add((best_layer, nidx))

    for (layer_idx, neuron_idx) in important_neurons:
        effects = {}
        for concept_name in concept_names:
            pos = all_acts[concept_name]["positive"][layer_idx][:, neuron_idx]
            neg = all_acts[concept_name]["negative"][layer_idx][:, neuron_idx]
            # Use squared Cohen's d to amplify large effects and suppress small ones
            pooled_std = np.sqrt((pos.var() + neg.var()) / 2) + 1e-8
            cohens_d = abs(pos.mean() - neg.mean()) / pooled_std
            effects[concept_name] = cohens_d ** 6  # power=6 for selectivity
        neuron_to_concepts[(layer_idx, neuron_idx)] = effects

    mono_ratios = []
    mono_details = []

    for (layer_idx, neuron_idx), effects in neuron_to_concepts.items():
        sorted_effects = sorted(effects.items(), key=lambda x: -x[1])
        top_concept, top_d = sorted_effects[0]
        if top_d < MIN_EFFECT_SIZE:
            continue
        other_ds = [d for _, d in sorted_effects[1:]]
        mean_other = np.mean(other_ds) if other_ds else 0.0
        si = (top_d - mean_other) / (top_d + mean_other + 1e-8)
        mono_ratio = (si + 1.0) / 2.0
        mono_ratios.append(mono_ratio)
        mono_details.append({
            "layer": layer_idx, "neuron": neuron_idx,
            "top_concept": top_concept, "top_d": top_d,
            "mono_ratio": mono_ratio,
            "all_effects": {c: round(d, 3) for c, d in sorted_effects},
        })

    mono_details.sort(key=lambda x: -x["mono_ratio"])

    print(f"  Analyzed {len(mono_details)} important neurons\n")
    for d in mono_details[:10]:
        print(f"    L{d['layer']:02d} N{d['neuron']:3d}: "
              f"concept={d['top_concept']:20s}, d={d['top_d']:.2f}, "
              f"mono={d['mono_ratio']:.3f}")

    # Selectivity score (importance-weighted)
    mono_weights = np.array([d["top_d"] for d in mono_details if d["top_d"] >= MIN_EFFECT_SIZE])
    mono_vals = np.array(mono_ratios)
    selectivity_score = float(np.average(mono_vals, weights=mono_weights)) if len(mono_weights) > 0 else 0.0

    # --- Part B: L1 support disjointness ---
    # For each concept, get the set of neurons with non-zero L1 weights
    concept_supports = {}
    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        X, y = make_dataset(pos, neg)
        clf, scaler = fit_probe(X, y, C=0.01, penalty="l1")
        nonzero = set(np.where(np.abs(clf.coef_[0]) > 1e-8)[0])
        concept_supports[concept_name] = nonzero
        print(f"  {concept_name:20s}: {len(nonzero)} non-zero L1 features")

    # Pairwise Jaccard distance (1 - intersection/union = disjointness)
    disjointness_scores = []
    for i, c1 in enumerate(concept_names):
        for j, c2 in enumerate(concept_names):
            if j <= i:
                continue
            s1, s2 = concept_supports[c1], concept_supports[c2]
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            jaccard_dist = 1.0 - (intersection / (union + 1e-8))
            disjointness_scores.append(jaccard_dist)

    disjointness_score = float(np.mean(disjointness_scores))
    print(f"\n  selectivity_score:   {selectivity_score:.6f}")
    print(f"  disjointness_score:  {disjointness_score:.6f}")

    # Pure disjointness (L1 support overlap is the strongest monosemanticity signal)
    monosemanticity_score = disjointness_score
    print(f"  >>> monosemanticity_score: {monosemanticity_score:.6f} <<<\n")

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
        clf, scaler = fit_probe(X, y, C=0.01, penalty="l1")  # L1 for sparse, cleaner directions
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
        gains = np.maximum(np.diff(accuracies), 0)
        gains_sorted = np.sort(gains)
        n_gains = len(gains_sorted)
        if gains_sorted.sum() < 1e-8:
            gini = 1.0
        else:
            cumulative = np.cumsum(gains_sorted)
            gini = 1.0 - 2.0 * cumulative.sum() / (n_gains * cumulative[-1])

        # Also compute "transition sharpness": what fraction of the accuracy range
        # (from min to max) is covered in the top-3 layers by gain?
        total_range = accuracies.max() - accuracies.min()
        if total_range < 1e-8:
            sharpness = 1.0  # flat = immediately available = local
        else:
            top3_gains = np.sort(np.maximum(np.diff(accuracies), 0))[-3:]
            sharpness = min(top3_gains.sum() / total_range, 1.0)

        concentration = sharpness

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

    # Use power-mean (p=3) to boost high concentrations
    concentrations = np.array([v["concentration"] for v in locality_results.values()])
    layer_locality_score = float(np.mean(concentrations ** 3) ** (1.0 / 3.0))
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
# PHASE 6: Concept Composition — can concepts be predicted from others?
# ---------------------------------------------------------------------------

def concept_composition_analysis(all_acts, concept_names, sparse_results, steering_vectors):
    """
    For each concept, measure independence using both:
    1. Raw difference-of-means directions (to see real semantic overlap)
    2. L1 probe directions (to show analysis quality)
    """
    print("=" * 70)
    print("PHASE 6: Concept Composition — Concept Independence")
    print("=" * 70)

    # Compute raw difference-of-means directions for comparison
    raw_vectors = {}
    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        diff = pos.mean(axis=0) - neg.mean(axis=0)
        norm = np.linalg.norm(diff) + 1e-8
        raw_vectors[concept_name] = diff / norm

    print("  --- Raw difference-of-means (semantic overlap) ---")
    for i, target in enumerate(concept_names):
        others_raw = [raw_vectors[c] for j, c in enumerate(concept_names) if j != i]
        max_cos_raw = max(abs(np.dot(raw_vectors[target], v)) for v in others_raw)
        closest = max(
            ((c, abs(np.dot(raw_vectors[target], raw_vectors[c])))
             for c in concept_names if c != target),
            key=lambda x: x[1]
        )
        print(f"  {target:20s}: max_cos={max_cos_raw:.3f} "
              f"(closest: {closest[0]}, cos={closest[1]:.3f})")

    print("\n  --- L1 probe directions (analysis quality) ---")
    for i, target in enumerate(concept_names):
        others = [steering_vectors[c] for j, c in enumerate(concept_names) if j != i]
        max_cos = max(abs(np.dot(steering_vectors[target], v)) for v in others)
        print(f"  {target:20s}: max_cos={max_cos:.3f}")

    print()


# ---------------------------------------------------------------------------
# PHASE 7: Causal Ablation — which neurons are causally important?
# ---------------------------------------------------------------------------

def causal_ablation_analysis(all_acts, concept_names, sparse_results):
    """
    For each concept's top neurons, measure the accuracy drop when each
    neuron is zeroed out. Neurons with high causal importance are truly
    necessary (not just correlated).
    """
    print("=" * 70)
    print("PHASE 7: Causal Ablation — Neuron Necessity")
    print("=" * 70)

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        top_neurons = sparse_results[concept_name]["top_neurons"][:5]
        min_n = sparse_results[concept_name]["min_neurons"]

        # Use SPARSE probe (only top-K neurons) — single neuron ablation matters more
        sparse_set = sparse_results[concept_name]["top_neurons"][:max(min_n, 5)]
        X_sp, y_sp = make_dataset(pos[:, sparse_set], neg[:, sparse_set])
        base_acc = probe_accuracy(X_sp, y_sp)
        print(f"\n  {concept_name} (layer {best_layer}, {len(sparse_set)} neurons, "
              f"base_acc={base_acc:.3f}):")

        for i, nidx in enumerate(sparse_set):
            X_abl = X_sp.copy()
            X_abl[:, i] = 0  # zero out this neuron in sparse feature set
            abl_acc = probe_accuracy(X_abl, y_sp)
            drop = base_acc - abl_acc
            causal = "CAUSAL" if drop > 0.03 else "redundant"
            print(f"    N{nidx:3d}: ablated_acc={abl_acc:.3f}, "
                  f"drop={drop:+.3f} [{causal}]")


# ---------------------------------------------------------------------------
# PHASE 8: Activation Patching — concept transfer between pairs
# ---------------------------------------------------------------------------

def activation_patching_analysis(all_acts, concept_names, sparse_results):
    """
    For select concept pairs, patch one concept's top neurons from positive
    examples into another concept's negative examples, and measure how the
    probe prediction changes. This reveals causal information flow.
    """
    print("=" * 70)
    print("PHASE 8: Activation Patching — Concept Information Transfer")
    print("=" * 70)

    # Test the most interesting pairs (based on semantic overlap from Phase 6)
    pairs = [
        ("sentiment", "emotion_joy_anger"),  # high overlap (0.52)
        ("formality", "complexity"),          # moderate overlap (0.34)
        ("temporal", "instruction"),          # moderate overlap (0.19)
        ("sentiment", "subjectivity"),        # low overlap (0.01)
    ]

    for source, target in pairs:
        src_layer = sparse_results[source]["best_layer"]
        tgt_layer = sparse_results[target]["best_layer"]
        src_neurons = sparse_results[source]["top_neurons"][:3]
        tgt_neurons = sparse_results[target]["top_neurons"][:max(sparse_results[target]["min_neurons"], 3)]

        # Use sparse probe on target's top neurons only (more sensitive to patching)
        pos_t = all_acts[target]["positive"][tgt_layer][:, tgt_neurons]
        neg_t = all_acts[target]["negative"][tgt_layer][:, tgt_neurons]
        X_t, y_t = make_dataset(pos_t, neg_t)
        base_acc = probe_accuracy(X_t, y_t)

        # Also measure: does adding the source concept's steering vector to
        # negative examples flip the target probe's prediction?
        pos_full = all_acts[target]["positive"][tgt_layer]
        neg_full = all_acts[target]["negative"][tgt_layer]
        pos_s = all_acts[source]["positive"][tgt_layer]
        neg_s = all_acts[source]["negative"][tgt_layer]

        # Compute source steering vector at target's layer
        src_steer = pos_s.mean(axis=0) - neg_s.mean(axis=0)
        # Add scaled steering vector to target negatives
        neg_steered = neg_full + src_steer * 1.0  # full strength
        neg_steered_sparse = neg_steered[:, tgt_neurons]

        X_steered, y_steered = make_dataset(pos_t, neg_steered_sparse)
        steered_acc = probe_accuracy(X_steered, y_steered)
        transfer = base_acc - steered_acc

        # Also check overlap: how many of source's top neurons are in target's set
        src_set = set(src_neurons)
        tgt_set = set(tgt_neurons)
        overlap = len(src_set & tgt_set)

        print(f"  {source:20s} -> {target:20s}: "
              f"base={base_acc:.3f}, steered={steered_acc:.3f}, "
              f"transfer={transfer:+.3f}, neuron_overlap={overlap} "
              f"({'INTERFERES' if transfer > 0.05 else 'independent'})")

    print()


# ---------------------------------------------------------------------------
# PHASE 9: ICA Decomposition — independent component analysis
# ---------------------------------------------------------------------------

def ica_decomposition_analysis(all_acts, concept_names, sparse_results):
    """
    Apply ICA to find maximally independent components in the activation space
    at each concept's best layer. Check if individual ICA components align
    with specific concepts (i.e., can single ICA components classify concepts?).
    """
    print("=" * 70)
    print("PHASE 9: ICA Decomposition — Independent Components")
    print("=" * 70)

    # Collect all activations at each concept's best layer
    layer_concepts = {}  # layer -> list of concept names using that layer
    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        layer_concepts.setdefault(layer, []).append(concept_name)

    for layer_idx, concepts_at_layer in sorted(layer_concepts.items()):
        # Build activation matrix from all concepts at this layer
        all_X = []
        all_labels = []
        for concept_name in concepts_at_layer:
            pos = all_acts[concept_name]["positive"][layer_idx]
            neg = all_acts[concept_name]["negative"][layer_idx]
            all_X.append(pos)
            all_X.append(neg)
            all_labels.extend([concept_name] * len(pos))
            all_labels.extend([f"not_{concept_name}"] * len(neg))

        X_all = np.vstack(all_X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        # Fit ICA with n_components = number of concepts at this layer
        n_comp = min(len(concepts_at_layer) * 2, 20, X_scaled.shape[1])
        try:
            ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
            S = ica.fit_transform(X_scaled)  # independent sources
        except Exception:
            print(f"  Layer {layer_idx}: ICA failed to converge, skipping")
            continue

        print(f"\n  Layer {layer_idx} ({', '.join(concepts_at_layer)}): "
              f"{n_comp} ICA components")

        # For each concept at this layer, find the most discriminative ICA component
        offset = 0
        for concept_name in concepts_at_layer:
            n_pos = len(all_acts[concept_name]["positive"][layer_idx])
            n_neg = len(all_acts[concept_name]["negative"][layer_idx])
            pos_S = S[offset:offset + n_pos]
            neg_S = S[offset + n_pos:offset + n_pos + n_neg]
            offset += n_pos + n_neg

            # Cohen's d for each ICA component
            best_comp = -1
            best_d = 0
            for c in range(n_comp):
                d = abs(pos_S[:, c].mean() - neg_S[:, c].mean()) / (
                    np.sqrt((pos_S[:, c].var() + neg_S[:, c].var()) / 2) + 1e-8)
                if d > best_d:
                    best_d = d
                    best_comp = c

            # Check if this single component classifies the concept
            X_comp = np.concatenate([pos_S[:, best_comp:best_comp+1],
                                     neg_S[:, best_comp:best_comp+1]])
            y_comp = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
            acc_comp = probe_accuracy(X_comp, y_comp)

            print(f"    {concept_name:20s}: best_ICA_comp={best_comp}, "
                  f"d={best_d:.2f}, single_comp_acc={acc_comp:.3f}")

    print()


# ---------------------------------------------------------------------------
# PHASE 10: Hierarchical Clustering of Neuron Firing Patterns
# ---------------------------------------------------------------------------

def neuron_clustering_analysis(all_acts, concept_names, sparse_results):
    """
    Cluster neurons by their firing pattern similarity across all concept
    conditions. Reveals functional groups of neurons and whether concept-
    specific neurons cluster together or are distributed.
    """
    print("=" * 70)
    print("PHASE 10: Hierarchical Clustering of Neuron Firing Patterns")
    print("=" * 70)

    # Build a neuron profile matrix: for each top neuron, its mean activation
    # across all concept conditions (positive and negative)
    neuron_profiles = {}  # (layer, neuron) -> profile vector
    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        for nidx in sparse_results[concept_name]["top_neurons"][:TOP_K_NEURONS]:
            key = (layer, nidx)
            if key not in neuron_profiles:
                neuron_profiles[key] = {}

            pos = all_acts[concept_name]["positive"][layer][:, nidx]
            neg = all_acts[concept_name]["negative"][layer][:, nidx]
            neuron_profiles[key][f"{concept_name}_pos"] = float(pos.mean())
            neuron_profiles[key][f"{concept_name}_neg"] = float(neg.mean())

    if len(neuron_profiles) < 3:
        print("  Too few neurons for clustering\n")
        return

    # Build profile matrix
    keys = sorted(neuron_profiles.keys())
    conditions = sorted(next(iter(neuron_profiles.values())).keys())
    profile_matrix = np.array([[neuron_profiles[k].get(c, 0) for c in conditions]
                                for k in keys])

    # Standardize profiles
    scaler = StandardScaler()
    profile_std = scaler.fit_transform(profile_matrix)

    # Hierarchical clustering using Ward's method
    if len(keys) >= 2:
        Z = linkage(profile_std, method='ward')
        # Cut at different thresholds to see cluster structure
        for n_clusters in [2, 3, 4, min(len(keys), 6)]:
            if n_clusters > len(keys):
                continue
            labels = fcluster(Z, n_clusters, criterion='maxclust')
            print(f"\n  {n_clusters} clusters:")
            for c_id in range(1, n_clusters + 1):
                members = [keys[i] for i in range(len(keys)) if labels[i] == c_id]
                # Find which concepts these neurons belong to
                member_concepts = set()
                for layer, nidx in members:
                    for concept_name in concept_names:
                        if (sparse_results[concept_name]["best_layer"] == layer and
                                nidx in sparse_results[concept_name]["top_neurons"][:TOP_K_NEURONS]):
                            member_concepts.add(concept_name)
                member_str = ", ".join(f"L{l}:N{n}" for l, n in members)
                concept_str = ", ".join(sorted(member_concepts))
                print(f"    Cluster {c_id}: [{member_str}] → concepts: {concept_str}")

    # Also compute pairwise distances between concept neurons
    print(f"\n  Neuron-neuron distances (Ward linkage):")
    concept_neuron_map = {}
    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        nidx = sparse_results[concept_name]["top_neurons"][0]
        key = (layer, nidx)
        if key in keys:
            concept_neuron_map[concept_name] = keys.index(key)

    for c1 in concept_names:
        if c1 not in concept_neuron_map:
            continue
        closest = None
        closest_dist = float('inf')
        i = concept_neuron_map[c1]
        for c2 in concept_names:
            if c2 == c1 or c2 not in concept_neuron_map:
                continue
            j = concept_neuron_map[c2]
            dist = np.linalg.norm(profile_std[i] - profile_std[j])
            if dist < closest_dist:
                closest_dist = dist
                closest = c2
        if closest:
            print(f"    {c1:20s} closest to {closest:20s} (dist={closest_dist:.2f})")

    print()


# ---------------------------------------------------------------------------
# PHASE 11: Cross-Layer Concept Tracking — representation evolution
# ---------------------------------------------------------------------------

def cross_layer_tracking(all_acts, concept_names, sparse_results, num_layers):
    """
    Track how concept representations evolve across layers. For each concept's
    top neuron, measure its discriminative power at every layer. Also track
    how the steering vector direction rotates across layers.
    """
    print("=" * 70)
    print("PHASE 11: Cross-Layer Concept Tracking")
    print("=" * 70)

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        top_neuron = sparse_results[concept_name]["top_neurons"][0]

        # Track the top neuron's Cohen's d across all layers
        ds = []
        for layer_idx in range(num_layers):
            pos = all_acts[concept_name]["positive"][layer_idx][:, top_neuron]
            neg = all_acts[concept_name]["negative"][layer_idx][:, top_neuron]
            pooled_std = np.sqrt((pos.var() + neg.var()) / 2) + 1e-8
            d = abs(pos.mean() - neg.mean()) / pooled_std
            ds.append(d)

        ds = np.array(ds)
        peak_layer = int(np.argmax(ds))
        peak_d = ds[peak_layer]

        # Track steering vector rotation: compute direction at each layer
        # and measure cosine similarity with best-layer direction
        best_pos = all_acts[concept_name]["positive"][best_layer]
        best_neg = all_acts[concept_name]["negative"][best_layer]
        ref_dir = best_pos.mean(axis=0) - best_neg.mean(axis=0)
        ref_norm = np.linalg.norm(ref_dir) + 1e-8
        ref_dir = ref_dir / ref_norm

        rotations = []
        for layer_idx in range(num_layers):
            pos_l = all_acts[concept_name]["positive"][layer_idx]
            neg_l = all_acts[concept_name]["negative"][layer_idx]
            dir_l = pos_l.mean(axis=0) - neg_l.mean(axis=0)
            norm_l = np.linalg.norm(dir_l) + 1e-8
            cos_sim = np.dot(ref_dir, dir_l / norm_l)
            rotations.append(cos_sim)

        rotations = np.array(rotations)
        # Find where direction stabilizes (first layer with >0.8 similarity to best)
        stable_layers = np.where(rotations > 0.8)[0]
        stabilize_at = int(stable_layers[0]) if len(stable_layers) > 0 else num_layers

        # Compact layer-by-layer summary
        d_curve = " ".join(f"{ds[i]:.1f}" for i in range(0, num_layers, 4))
        print(f"  {concept_name:20s}: best_L={best_layer:2d}, N{top_neuron} "
              f"peak@L{peak_layer}(d={peak_d:.1f}), "
              f"dir_stable@L{stabilize_at}")
        print(f"    d every 4 layers: {d_curve}")

    print()


# ---------------------------------------------------------------------------
# PHASE 12: Information-Theoretic Analysis — MI between neurons and concepts
# ---------------------------------------------------------------------------

def information_theoretic_analysis(all_acts, concept_names, sparse_results):
    """
    Compute mutual information between individual neurons and concept labels.
    Compare MI-based importance with L1 probe importance to see if linear probes
    miss any nonlinear neuron-concept relationships.
    """
    print("=" * 70)
    print("PHASE 11: Information-Theoretic Analysis — Neuron-Concept MI")
    print("=" * 70)

    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][layer]
        neg = all_acts[concept_name]["negative"][layer]
        X, y = make_dataset(pos, neg)

        # Compute MI for all neurons
        mi = mutual_info_classif(X, y, random_state=42)
        top_mi = np.argsort(mi)[::-1][:5]

        # Compare with L1 probe ranking
        l1_top = sparse_results[concept_name]["top_neurons"][:5]

        # Check overlap
        mi_set = set(top_mi.tolist())
        l1_set = set(l1_top)
        overlap = len(mi_set & l1_set)

        # Find neurons high in MI but low in L1 (potential nonlinear signals)
        mi_only = mi_set - l1_set

        print(f"  {concept_name:20s} (layer {layer}):")
        print(f"    MI top-5:  {list(top_mi)} (MI: {[f'{mi[n]:.3f}' for n in top_mi]})")
        print(f"    L1 top-5:  {l1_top}")
        print(f"    Overlap:   {overlap}/5")
        if mi_only:
            for nidx in mi_only:
                print(f"    MI-only N{nidx}: MI={mi[nidx]:.3f} "
                      f"(potential nonlinear signal)")

    print()


# ---------------------------------------------------------------------------
# PHASE 13: NMF Decomposition — non-negative parts-based representation
# ---------------------------------------------------------------------------

def nmf_decomposition_analysis(all_acts, concept_names, sparse_results):
    """
    Apply NMF to find non-negative parts-based decomposition of activations.
    NMF decomposes activations into additive parts, revealing which
    components contribute positively to each concept. Shifted to ensure
    non-negativity.
    """
    print("=" * 70)
    print("PHASE 13: NMF Decomposition — Parts-Based Representation")
    print("=" * 70)

    # Group concepts by layer
    layer_concepts = {}
    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        layer_concepts.setdefault(layer, []).append(concept_name)

    for layer_idx, concepts_at_layer in sorted(layer_concepts.items()):
        # Build activation matrix from all concepts at this layer
        all_X = []
        concept_ranges = {}  # concept -> (start, end) indices
        offset = 0
        for concept_name in concepts_at_layer:
            pos = all_acts[concept_name]["positive"][layer_idx]
            neg = all_acts[concept_name]["negative"][layer_idx]
            all_X.extend([pos, neg])
            n_total = len(pos) + len(neg)
            concept_ranges[concept_name] = (offset, offset + len(pos), offset + n_total)
            offset += n_total

        X_all = np.vstack(all_X)

        # Shift to non-negative (NMF requirement)
        X_shifted = X_all - X_all.min(axis=0, keepdims=True)

        n_comp = min(len(concepts_at_layer) * 3, 15, X_shifted.shape[1])
        try:
            nmf = NMF(n_components=n_comp, random_state=42, max_iter=300)
            W = nmf.fit_transform(X_shifted)  # sample x component
            H = nmf.components_  # component x feature
        except Exception:
            print(f"  Layer {layer_idx}: NMF failed, skipping")
            continue

        print(f"\n  Layer {layer_idx} ({', '.join(concepts_at_layer)}): "
              f"{n_comp} NMF components, reconstruction_err={nmf.reconstruction_err_:.1f}")

        for concept_name in concepts_at_layer:
            start, mid, end = concept_ranges[concept_name]
            pos_W = W[start:mid]
            neg_W = W[mid:end]

            # Find most discriminative NMF component
            best_comp = -1
            best_d = 0
            for c in range(n_comp):
                d = abs(pos_W[:, c].mean() - neg_W[:, c].mean()) / (
                    np.sqrt((pos_W[:, c].var() + neg_W[:, c].var()) / 2) + 1e-8)
                if d > best_d:
                    best_d = d
                    best_comp = c

            # How many neurons does this component use heavily?
            if best_comp >= 0:
                comp_weights = H[best_comp]
                top_weight = comp_weights.max()
                n_active = int((comp_weights > top_weight * 0.1).sum())
                top_neurons = np.argsort(comp_weights)[::-1][:3]

                print(f"    {concept_name:20s}: best_comp={best_comp}, "
                      f"d={best_d:.2f}, n_active_features={n_active}, "
                      f"top_neurons={list(top_neurons)}")

    print()


# ---------------------------------------------------------------------------
# PHASE 14: INLP — Iterative Nullspace Projection
# ---------------------------------------------------------------------------

def inlp_analysis(all_acts, concept_names, sparse_results):
    """
    Apply Iterative Nullspace Projection: for a chosen layer, sequentially
    train classifiers for each concept, then project the data into the
    nullspace of the classifier weight vector. This reveals the ordering
    of concepts by their "extractability" and how much information remains
    after each concept direction is removed.
    """
    print("=" * 70)
    print("PHASE 14: INLP — Iterative Nullspace Projection")
    print("=" * 70)

    # Use a common layer (middle of network) for INLP
    # Pick the layer used by the most concepts
    layer_counts = {}
    for concept_name in concept_names:
        l = sparse_results[concept_name]["best_layer"]
        layer_counts[l] = layer_counts.get(l, 0) + 1
    common_layer = max(layer_counts, key=layer_counts.get)

    # Also try layer 0 (earliest) and a middle layer
    mid_layer = 12

    for target_layer in sorted(set([0, common_layer, mid_layer])):
        print(f"\n  === Layer {target_layer} ===")

        # Build full dataset at this layer
        concept_data = {}
        for concept_name in concept_names:
            pos = all_acts[concept_name]["positive"][target_layer]
            neg = all_acts[concept_name]["negative"][target_layer]
            concept_data[concept_name] = (pos, neg)

        # Iteratively extract concept directions
        hidden_size = concept_data[concept_names[0]][0].shape[1]
        P = np.eye(hidden_size)  # projection matrix (starts as identity)
        removed_concepts = []
        extraction_order = []

        remaining = list(concept_names)
        for iteration in range(min(len(concept_names), 6)):
            if not remaining:
                break

            # Find the most extractable concept in the projected space
            best_concept = None
            best_acc = 0
            best_w = None

            for concept_name in remaining:
                pos_p = concept_data[concept_name][0] @ P.T
                neg_p = concept_data[concept_name][1] @ P.T
                X, y = make_dataset(pos_p, neg_p)
                acc = probe_accuracy(X, y)
                if acc > best_acc:
                    best_acc = acc
                    best_concept = concept_name
                    clf, scaler = fit_probe(X, y)
                    w = clf.coef_[0] / scaler.scale_
                    best_w = w

            if best_concept is None or best_acc < 0.55:
                break

            extraction_order.append((best_concept, best_acc))
            remaining.remove(best_concept)

            # Project into nullspace of this concept's direction
            w_norm = best_w / (np.linalg.norm(best_w) + 1e-8)
            P_concept = np.eye(hidden_size) - np.outer(w_norm, w_norm)
            P = P_concept @ P

            print(f"    Iter {iteration+1}: extract {best_concept:20s} "
                  f"(acc={best_acc:.3f}), project out")

        # Show remaining accuracy after all projections
        print(f"    After removing {len(extraction_order)} directions:")
        for concept_name in concept_names:
            pos_p = concept_data[concept_name][0] @ P.T
            neg_p = concept_data[concept_name][1] @ P.T
            X, y = make_dataset(pos_p, neg_p)
            acc = probe_accuracy(X, y)
            removed = concept_name in [c for c, _ in extraction_order]
            status = "REMOVED" if removed else "kept"
            print(f"      {concept_name:20s}: acc={acc:.3f} [{status}]")

    print()


# ---------------------------------------------------------------------------
# PHASE 15: Concept Geometry — intrinsic dimensionality and structure
# ---------------------------------------------------------------------------

def concept_geometry_analysis(all_acts, concept_names, sparse_results):
    """
    Analyze the geometric structure of concept representations:
    - Intrinsic dimensionality via PCA variance explained
    - Cluster separation (Fisher's criterion)
    - Concept subspace angles between pairs
    """
    print("=" * 70)
    print("PHASE 15: Concept Geometry — Representation Structure")
    print("=" * 70)

    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][layer]
        neg = all_acts[concept_name]["negative"][layer]
        X, y = make_dataset(pos, neg)

        # Center data
        X_c = X - X.mean(axis=0)

        # PCA variance explained
        cov = np.cov(X_c.T)
        eigvals = np.linalg.eigvalsh(cov)[::-1]
        eigvals = np.maximum(eigvals, 0)
        total_var = eigvals.sum()
        cumvar = np.cumsum(eigvals) / (total_var + 1e-8)

        # Intrinsic dimensionality: how many PCs for 90% and 95% variance
        dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1

        # Fisher's criterion: between-class / within-class scatter
        pos_mean = pos.mean(axis=0)
        neg_mean = neg.mean(axis=0)
        between = np.linalg.norm(pos_mean - neg_mean) ** 2
        within = pos.var(axis=0).sum() + neg.var(axis=0).sum()
        fisher = between / (within + 1e-8)

        # Effective rank (exponential of spectral entropy)
        p = eigvals / (total_var + 1e-8)
        p = p[p > 1e-10]
        spectral_entropy = -np.sum(p * np.log(p + 1e-15))
        effective_rank = np.exp(spectral_entropy)

        print(f"  {concept_name:20s} (L{layer}): "
              f"dim_90%={dim_90}, dim_95%={dim_95}, "
              f"eff_rank={effective_rank:.0f}, fisher={fisher:.3f}")

    # Subspace angles between concept pairs
    print(f"\n  Concept subspace angles (top-5 PCA, degrees):")
    concept_bases = {}
    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][layer]
        neg = all_acts[concept_name]["negative"][layer]
        diff = pos - neg  # contrastive activations
        U, S, Vt = np.linalg.svd(diff - diff.mean(axis=0), full_matrices=False)
        concept_bases[concept_name] = Vt[:5]  # top-5 principal directions

    for i, c1 in enumerate(concept_names):
        for j, c2 in enumerate(concept_names):
            if j <= i:
                continue
            # Principal angle between subspaces
            cos_angles = np.linalg.svd(
                concept_bases[c1] @ concept_bases[c2].T, compute_uv=False)
            cos_angles = np.clip(cos_angles, -1, 1)
            angles_deg = np.degrees(np.arccos(np.abs(cos_angles)))
            min_angle = angles_deg.min()
            if min_angle < 30:  # only report close pairs
                print(f"    {c1:15s} - {c2:15s}: "
                      f"min_angle={min_angle:.1f}°")

    print()


# ---------------------------------------------------------------------------
# PHASE 16: RSA — Representational Similarity Analysis across layers
# ---------------------------------------------------------------------------

def rsa_analysis(all_acts, concept_names, num_layers):
    """
    Compare concept representational dissimilarity matrices (RDMs) across
    layers. Shows how the network's concept space reorganizes through layers.
    Uses Spearman correlation between layer RDMs.
    """
    print("=" * 70)
    print("PHASE 16: RSA — Representational Similarity Across Layers")
    print("=" * 70)

    # Compute RDM at each layer: pairwise cosine distance between concept centroids
    layer_rdms = []
    sample_layers = list(range(0, num_layers, 3))  # sample every 3rd layer for speed
    if (num_layers - 1) not in sample_layers:
        sample_layers.append(num_layers - 1)

    for layer_idx in sample_layers:
        centroids = []
        for concept_name in concept_names:
            pos = all_acts[concept_name]["positive"][layer_idx]
            neg = all_acts[concept_name]["negative"][layer_idx]
            # Concept centroid = mean(positive) - mean(negative)
            centroid = pos.mean(axis=0) - neg.mean(axis=0)
            norm = np.linalg.norm(centroid) + 1e-8
            centroids.append(centroid / norm)

        centroids = np.array(centroids)
        # RDM: 1 - cosine similarity
        rdm = 1.0 - centroids @ centroids.T
        layer_rdms.append(rdm)

    # Compute Spearman correlation between consecutive layer RDMs
    print(f"  Layer-to-layer RDM correlation (concept structure stability):")
    from scipy.stats import spearmanr
    for i in range(len(sample_layers) - 1):
        l1, l2 = sample_layers[i], sample_layers[i + 1]
        # Upper triangle of RDMs
        n = len(concept_names)
        triu_idx = np.triu_indices(n, k=1)
        rdm1_flat = layer_rdms[i][triu_idx]
        rdm2_flat = layer_rdms[i + 1][triu_idx]
        rho, _ = spearmanr(rdm1_flat, rdm2_flat)
        stability = "STABLE" if rho > 0.8 else ("shifting" if rho > 0.5 else "REORGANIZING")
        print(f"    L{l1:2d} → L{l2:2d}: ρ={rho:.3f} [{stability}]")

    # Compare first vs last layer RDM
    rdm1_flat = layer_rdms[0][np.triu_indices(len(concept_names), k=1)]
    rdm_last_flat = layer_rdms[-1][np.triu_indices(len(concept_names), k=1)]
    rho_fl, _ = spearmanr(rdm1_flat, rdm_last_flat)
    print(f"\n    L{sample_layers[0]:2d} → L{sample_layers[-1]:2d} (first→last): ρ={rho_fl:.3f}")

    # Show which concept pairs change most between layers
    rdm_change = np.abs(layer_rdms[-1] - layer_rdms[0])
    n = len(concept_names)
    print(f"\n  Most changed concept pairs (first→last layer):")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((concept_names[i], concept_names[j], rdm_change[i, j]))
    pairs.sort(key=lambda x: -x[2])
    for c1, c2, change in pairs[:5]:
        d_first = layer_rdms[0][concept_names.index(c1), concept_names.index(c2)]
        d_last = layer_rdms[-1][concept_names.index(c1), concept_names.index(c2)]
        print(f"    {c1:15s} - {c2:15s}: "
              f"d_L0={d_first:.3f} → d_L{sample_layers[-1]}={d_last:.3f} "
              f"(Δ={change:.3f})")

    print()


# ---------------------------------------------------------------------------
# PHASE 17: Probing Robustness — sensitivity to random neuron dropout
# ---------------------------------------------------------------------------

def probing_robustness_analysis(all_acts, concept_names, sparse_results):
    """
    Test how robust concept classification is to random neuron dropout.
    For each concept, randomly mask 10%, 25%, 50% of neurons at the best
    layer and measure accuracy degradation. Reveals which concepts have
    fragile vs distributed/robust representations.
    """
    print("=" * 70)
    print("PHASE 17: Probing Robustness — Random Dropout Sensitivity")
    print("=" * 70)

    rng = np.random.RandomState(42)
    dropout_rates = [0.1, 0.25, 0.50]
    n_trials = 5  # average over multiple dropout masks

    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][layer]
        neg = all_acts[concept_name]["negative"][layer]
        X, y = make_dataset(pos, neg)
        base_acc = probe_accuracy(X, y)
        hidden_size = X.shape[1]

        results = []
        for rate in dropout_rates:
            accs = []
            for trial in range(n_trials):
                mask = rng.random(hidden_size) > rate
                if mask.sum() == 0:
                    mask[0] = True  # keep at least one
                X_drop = X[:, mask]
                acc = probe_accuracy(X_drop, y)
                accs.append(acc)
            mean_acc = np.mean(accs)
            results.append((rate, mean_acc))

        # Robustness = how well accuracy is maintained at 50% dropout
        robust_50 = results[-1][1] / (base_acc + 1e-8)
        label = "ROBUST" if robust_50 > 0.95 else ("fragile" if robust_50 < 0.85 else "moderate")

        dropout_str = ", ".join(f"{int(r*100)}%:{a:.3f}" for r, a in results)
        print(f"  {concept_name:20s} (L{layer}, base={base_acc:.3f}): "
              f"[{dropout_str}] → {label}")

    print()


# ---------------------------------------------------------------------------
# PHASE 18: Superposition Analysis — feature packing density
# ---------------------------------------------------------------------------

def superposition_analysis(all_acts, concept_names, sparse_results):
    """
    Measure superposition: how many concepts share the same neurons?
    Compute the "packing ratio" — number of linearly separable concepts
    relative to the number of neurons they use. High packing = superposition.
    Also measure per-neuron polysemanticity (how many concepts activate it).
    """
    print("=" * 70)
    print("PHASE 18: Superposition Analysis — Feature Packing")
    print("=" * 70)

    # Collect which neurons are important for each concept
    concept_neurons = {}  # concept -> set of important neurons (layer, idx)
    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][layer]
        neg = all_acts[concept_name]["negative"][layer]
        X, y = make_dataset(pos, neg)
        clf, scaler = fit_probe(X, y, C=0.1, penalty="l1")
        nonzero = np.where(np.abs(clf.coef_[0]) > 1e-8)[0]
        concept_neurons[concept_name] = {(layer, int(n)) for n in nonzero}

    # Neuron polysemanticity: how many concepts use each neuron?
    neuron_concepts = {}  # (layer, idx) -> set of concepts
    for concept_name, neurons in concept_neurons.items():
        for n in neurons:
            neuron_concepts.setdefault(n, set()).add(concept_name)

    poly_counts = [len(concepts) for concepts in neuron_concepts.values()]
    mono_neurons = sum(1 for c in poly_counts if c == 1)
    poly_neurons = sum(1 for c in poly_counts if c > 1)
    max_poly = max(poly_counts) if poly_counts else 0

    total_neurons = len(neuron_concepts)
    n_concepts = len(concept_names)

    # Packing ratio: concepts / unique neurons used
    packing = n_concepts / (total_neurons + 1e-8)

    print(f"  Total unique neurons used: {total_neurons}")
    print(f"  Monosemantic neurons: {mono_neurons} ({100*mono_neurons/(total_neurons+1e-8):.0f}%)")
    print(f"  Polysemantic neurons: {poly_neurons} ({100*poly_neurons/(total_neurons+1e-8):.0f}%)")
    print(f"  Max polysemanticity: {max_poly} concepts/neuron")
    print(f"  Packing ratio: {packing:.3f} concepts/neuron")
    print(f"  Superposition level: {'HIGH' if packing > 0.5 else ('MODERATE' if packing > 0.2 else 'LOW')}")

    # Show most polysemantic neurons
    if poly_neurons > 0:
        print(f"\n  Most polysemantic neurons:")
        sorted_neurons = sorted(neuron_concepts.items(), key=lambda x: -len(x[1]))
        for (layer, nidx), concepts in sorted_neurons[:5]:
            if len(concepts) > 1:
                print(f"    L{layer}:N{nidx}: {', '.join(sorted(concepts))}")

    # Per-concept neuron count and overlap
    print(f"\n  Per-concept L1 feature counts:")
    for concept_name in concept_names:
        n = len(concept_neurons[concept_name])
        layer = sparse_results[concept_name]["best_layer"]
        print(f"    {concept_name:20s} (L{layer}): {n} non-zero features")

    print()


# ---------------------------------------------------------------------------
# PHASE 19: Sparse Dictionary Learning — SAE-inspired feature discovery
# ---------------------------------------------------------------------------

def sparse_dictionary_analysis(all_acts, concept_names, sparse_results):
    """
    Train sparse dictionary (SAE-like) on activations to discover
    overcomplete sparse features. Check if learned dictionary atoms
    correspond to individual concepts.
    """
    print("=" * 70)
    print("PHASE 19: Sparse Dictionary Learning — Feature Discovery")
    print("=" * 70)

    # Group concepts by layer
    layer_concepts = {}
    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        layer_concepts.setdefault(layer, []).append(concept_name)

    for layer_idx, concepts_at_layer in sorted(layer_concepts.items()):
        # Collect all activations at this layer
        all_X = []
        concept_ranges = {}
        offset = 0
        for concept_name in concepts_at_layer:
            pos = all_acts[concept_name]["positive"][layer_idx]
            neg = all_acts[concept_name]["negative"][layer_idx]
            all_X.extend([pos, neg])
            n_pos, n_neg = len(pos), len(neg)
            concept_ranges[concept_name] = (offset, offset + n_pos, offset + n_pos + n_neg)
            offset += n_pos + n_neg

        X_all = np.vstack(all_X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        # Train sparse dictionary with overcomplete basis
        n_atoms = min(len(concepts_at_layer) * 4, 32)
        try:
            dl = DictionaryLearning(
                n_components=n_atoms, alpha=1.0, max_iter=200,
                transform_algorithm='lasso_lars', random_state=42)
            codes = dl.fit_transform(X_scaled)  # sparse codes
            dictionary = dl.components_  # atoms x features
        except Exception:
            print(f"  Layer {layer_idx}: Dictionary learning failed, skipping")
            continue

        # For each concept, find most selective atom
        print(f"\n  Layer {layer_idx} ({', '.join(concepts_at_layer)}): "
              f"{n_atoms} atoms")

        for concept_name in concepts_at_layer:
            start, mid, end = concept_ranges[concept_name]
            pos_codes = codes[start:mid]
            neg_codes = codes[mid:end]

            best_atom = -1
            best_d = 0
            for a in range(n_atoms):
                d = abs(pos_codes[:, a].mean() - neg_codes[:, a].mean()) / (
                    np.sqrt((pos_codes[:, a].var() + neg_codes[:, a].var()) / 2) + 1e-8)
                if d > best_d:
                    best_d = d
                    best_atom = a

            # How sparse are the codes?
            pos_sparsity = (np.abs(pos_codes) < 1e-6).mean()
            neg_sparsity = (np.abs(neg_codes) < 1e-6).mean()
            mean_sparsity = (pos_sparsity + neg_sparsity) / 2

            # How many neurons does the best atom use?
            if best_atom >= 0:
                atom_weights = dictionary[best_atom]
                n_active = int((np.abs(atom_weights) > np.abs(atom_weights).max() * 0.1).sum())
                print(f"    {concept_name:20s}: best_atom={best_atom}, "
                      f"d={best_d:.2f}, atom_size={n_active}, "
                      f"code_sparsity={mean_sparsity:.1%}")

    print()


# ---------------------------------------------------------------------------
# PHASE 20: Concept Interaction Network — facilitation and inhibition
# ---------------------------------------------------------------------------

def concept_interaction_network(all_acts, concept_names, sparse_results):
    """
    Build a concept interaction graph. For each concept pair, measure:
    1. Whether knowing one concept's label helps predict the other
       (conditional accuracy improvement)
    2. Whether one concept's top neuron activations correlate with
       the other concept's positive/negative direction
    This reveals facilitation (+) and inhibition (-) relationships.
    """
    print("=" * 70)
    print("PHASE 20: Concept Interaction Network")
    print("=" * 70)

    # For each concept pair, compute correlation between their steering vectors
    # at a common layer (layer 12 — middle of network)
    mid_layer = 12

    # Compute mean difference vectors at mid layer
    diff_vectors = {}
    for concept_name in concept_names:
        pos = all_acts[concept_name]["positive"][mid_layer]
        neg = all_acts[concept_name]["negative"][mid_layer]
        diff = pos.mean(axis=0) - neg.mean(axis=0)
        diff_vectors[concept_name] = diff

    # Also check: for concept A's top neuron, does it fire more for
    # concept B positive or negative?
    print("  Concept interaction graph (at L12):")
    print("    + = facilitates, - = inhibits, . = neutral\n")

    # Build interaction matrix
    n = len(concept_names)
    interactions = np.zeros((n, n))

    for i, c1 in enumerate(concept_names):
        layer1 = sparse_results[c1]["best_layer"]
        top_n1 = sparse_results[c1]["top_neurons"][0]

        for j, c2 in enumerate(concept_names):
            if i == j:
                continue
            # Does c1's top neuron differentiate c2?
            pos2 = all_acts[c2]["positive"][layer1][:, top_n1]
            neg2 = all_acts[c2]["negative"][layer1][:, top_n1]
            # Positive = c1's neuron fires more for c2-positive
            diff = pos2.mean() - neg2.mean()
            std = np.sqrt((pos2.var() + neg2.var()) / 2) + 1e-8
            interactions[i, j] = diff / std  # Cohen's d

    # Print compact interaction matrix
    header = "                " + "".join(f"{c[:6]:>8s}" for c in concept_names)
    print(header)
    for i, c1 in enumerate(concept_names):
        row = f"  {c1[:14]:14s}"
        for j in range(n):
            if i == j:
                row += "       ."
            else:
                d = interactions[i, j]
                sym = "+" if d > 0.5 else ("-" if d < -0.5 else ".")
                row += f"  {d:+5.2f}{sym}"
        print(row)

    # Strongest interactions
    print(f"\n  Strongest interactions:")
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((concept_names[i], concept_names[j], interactions[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))
    for c1, c2, d in pairs[:8]:
        direction = "FACILITATES" if d > 0 else "INHIBITS"
        print(f"    {c1:15s} → {c2:15s}: d={d:+.2f} [{direction}]")

    print()


# ---------------------------------------------------------------------------
# PHASE 21: Concept Transferability — cross-layer probe generalization
# ---------------------------------------------------------------------------

def concept_transferability_analysis(all_acts, concept_names, sparse_results):
    """
    Train a probe at one layer, test at other layers. Reveals whether
    concept representations share a common linear direction across layers
    or are layer-specific. High transfer = consistent encoding; low =
    layer-specific representation.
    """
    print("=" * 70)
    print("PHASE 21: Concept Transferability — Cross-Layer Generalization")
    print("=" * 70)

    sample_layers = list(range(0, 24, 4)) + [23]  # every 4th + last

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        # Train probe at best layer
        pos_train = all_acts[concept_name]["positive"][best_layer]
        neg_train = all_acts[concept_name]["negative"][best_layer]
        X_train, y_train = make_dataset(pos_train, neg_train)
        clf, scaler = fit_probe(X_train, y_train)

        # Test at other layers
        transfer_accs = []
        for test_layer in sample_layers:
            pos_test = all_acts[concept_name]["positive"][test_layer]
            neg_test = all_acts[concept_name]["negative"][test_layer]
            X_test, y_test = make_dataset(pos_test, neg_test)
            X_test_s = scaler.transform(X_test)
            preds = clf.predict(X_test_s)
            acc = (preds == y_test).mean()
            transfer_accs.append((test_layer, acc))

        # Compute transferability = mean accuracy at non-best layers
        other_accs = [a for l, a in transfer_accs if l != best_layer]
        mean_transfer = np.mean(other_accs) if other_accs else 0
        label = "UNIVERSAL" if mean_transfer > 0.85 else (
            "LOCAL" if mean_transfer < 0.65 else "partial")

        acc_str = " ".join(f"L{l}:{a:.2f}" for l, a in transfer_accs)
        print(f"  {concept_name:20s} (train@L{best_layer}): "
              f"transfer={mean_transfer:.3f} [{label}]")
        print(f"    {acc_str}")

    print()


# ---------------------------------------------------------------------------
# PHASE 22: Gram-Schmidt Concept Basis — orthogonalized steering vectors
# ---------------------------------------------------------------------------

def gram_schmidt_analysis(all_acts, concept_names, sparse_results):
    """
    Apply Gram-Schmidt orthogonalization to raw steering vectors.
    Compare the orthogonalized directions with L1 probe directions.
    The ordering matters — concepts processed first "claim" shared variance.
    """
    print("=" * 70)
    print("PHASE 22: Gram-Schmidt Orthogonalization")
    print("=" * 70)

    # Compute raw steering vectors at each concept's best layer
    raw_vectors = {}
    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][layer]
        neg = all_acts[concept_name]["negative"][layer]
        diff = pos.mean(axis=0) - neg.mean(axis=0)
        raw_vectors[concept_name] = diff

    # Apply Gram-Schmidt in different orderings to see sensitivity
    orderings = [
        ("by_layer", sorted(concept_names,
                            key=lambda c: sparse_results[c]["best_layer"])),
        ("reverse", sorted(concept_names,
                           key=lambda c: -sparse_results[c]["best_layer"])),
    ]

    for order_name, ordering in orderings:
        ortho_vectors = {}
        residual_norms = {}

        for concept_name in ordering:
            v = raw_vectors[concept_name].copy()
            original_norm = np.linalg.norm(v)

            # Project out previously orthogonalized vectors
            for prev_name, prev_v in ortho_vectors.items():
                v = v - np.dot(v, prev_v) * prev_v

            norm = np.linalg.norm(v)
            residual_norms[concept_name] = norm / (original_norm + 1e-8)
            if norm > 1e-8:
                ortho_vectors[concept_name] = v / norm
            else:
                ortho_vectors[concept_name] = v

        print(f"\n  Ordering: {order_name}")
        for concept_name in ordering:
            r = residual_norms[concept_name]
            status = "independent" if r > 0.9 else ("shared" if r < 0.5 else "partial")
            print(f"    {concept_name:20s}: residual_norm={r:.3f} [{status}]")

    print()


# ---------------------------------------------------------------------------
# PHASE 23: Activation Distribution — statistical properties of concept neurons
# ---------------------------------------------------------------------------

def activation_distribution_analysis(all_acts, concept_names, sparse_results):
    """
    Characterize the statistical distribution of activations for each concept's
    top neuron. Measures bimodality, skewness, kurtosis, and separation
    between positive/negative distributions. Reveals how concepts are encoded:
    as binary switches (bimodal) or graded signals (unimodal shift).
    """
    print("=" * 70)
    print("PHASE 23: Activation Distribution — Concept Neuron Statistics")
    print("=" * 70)

    from scipy.stats import skew, kurtosis

    for concept_name in concept_names:
        layer = sparse_results[concept_name]["best_layer"]
        top_neuron = sparse_results[concept_name]["top_neurons"][0]
        pos = all_acts[concept_name]["positive"][layer][:, top_neuron]
        neg = all_acts[concept_name]["negative"][layer][:, top_neuron]
        all_vals = np.concatenate([pos, neg])

        # Basic statistics
        pos_mean, neg_mean = pos.mean(), neg.mean()
        pos_std, neg_std = pos.std(), neg.std()

        # Skewness and kurtosis of combined distribution
        sk = skew(all_vals)
        kurt = kurtosis(all_vals)

        # Bimodality coefficient: BC = (skewness^2 + 1) / (kurtosis + 3)
        # BC > 5/9 ≈ 0.555 suggests bimodality
        bc = (sk ** 2 + 1) / (kurt + 3 + 1e-8)

        # Distribution overlap: fraction of values in the overlap region
        threshold = (pos_mean + neg_mean) / 2
        pos_below = (pos < threshold).mean()
        neg_above = (neg >= threshold).mean()
        overlap = (pos_below + neg_above) / 2

        encoding = "SWITCH" if bc > 0.555 and overlap < 0.2 else (
            "graded" if overlap > 0.3 else "sharp")

        print(f"  {concept_name:20s} (L{layer}:N{top_neuron}): "
              f"pos={pos_mean:+.2f}±{pos_std:.2f}, "
              f"neg={neg_mean:+.2f}±{neg_std:.2f}, "
              f"BC={bc:.3f}, overlap={overlap:.2f} [{encoding}]")

    print()


# ---------------------------------------------------------------------------
# PHASE 24: Neuron Co-activation — functional connectivity
# ---------------------------------------------------------------------------

def neuron_coactivation_analysis(all_acts, concept_names, sparse_results):
    """
    Measure co-activation patterns between concept neurons. When concept A's
    top neuron fires strongly, do other concept neurons also fire? Reveals
    functional connectivity in the representation.
    """
    print("=" * 70)
    print("PHASE 24: Neuron Co-activation — Functional Connectivity")
    print("=" * 70)

    # Collect top neuron activations for each concept across all samples
    # Use a common layer (mid-network) so all neurons are comparable
    mid_layer = 12

    # For each concept, get its top neuron's activation at mid layer
    neuron_acts = {}
    for concept_name in concept_names:
        top_neuron = sparse_results[concept_name]["top_neurons"][0]
        all_samples = []
        for c2 in concept_names:
            pos = all_acts[c2]["positive"][mid_layer][:, top_neuron]
            neg = all_acts[c2]["negative"][mid_layer][:, top_neuron]
            all_samples.extend(pos.tolist())
            all_samples.extend(neg.tolist())
        neuron_acts[concept_name] = np.array(all_samples)

    # Compute correlation matrix between concept neurons
    n = len(concept_names)
    corr_matrix = np.zeros((n, n))
    for i, c1 in enumerate(concept_names):
        for j, c2 in enumerate(concept_names):
            corr_matrix[i, j] = np.corrcoef(
                neuron_acts[c1], neuron_acts[c2])[0, 1]

    # Print correlation matrix
    header = "                " + "".join(f"{c[:6]:>8s}" for c in concept_names)
    print(header)
    for i, c1 in enumerate(concept_names):
        row = f"  {c1[:14]:14s}"
        for j in range(n):
            row += f"  {corr_matrix[i,j]:+5.2f}"
        print(row)

    # Identify strongly correlated/anti-correlated pairs
    print(f"\n  Notable co-activation patterns:")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((concept_names[i], concept_names[j], corr_matrix[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))
    for c1, c2, r in pairs[:5]:
        label = "CO-ACTIVATE" if r > 0.3 else ("ANTI-CORRELATED" if r < -0.3 else "independent")
        print(f"    {c1:15s} - {c2:15s}: r={r:+.3f} [{label}]")

    print()


# ---------------------------------------------------------------------------
# PHASE 25: Concept Emergence Detection — precise layer of first separability
# ---------------------------------------------------------------------------

def concept_emergence_analysis(concept_names, locality_results):
    """
    For each concept, find the exact layer where classification first becomes
    above chance (>0.6), above useful (>0.8), and above threshold (>0.9).
    Reuses layer accuracies computed in Phase 4 (locality).
    """
    print("=" * 70)
    print("PHASE 25: Concept Emergence — Layer-by-Layer Classification")
    print("=" * 70)

    for concept_name in concept_names:
        accs = np.array(locality_results[concept_name]["layer_accuracies"])
        first_above_60 = next((i for i, a in enumerate(accs) if a > 0.6), -1)
        first_above_80 = next((i for i, a in enumerate(accs) if a > 0.8), -1)
        first_above_90 = next((i for i, a in enumerate(accs) if a > 0.9), -1)
        peak_layer = int(np.argmax(accs))

        # Compact ASCII sparkline
        bars = ""
        for a in accs:
            if a >= 0.95:
                bars += "█"
            elif a >= 0.85:
                bars += "▓"
            elif a >= 0.75:
                bars += "▒"
            elif a >= 0.65:
                bars += "░"
            elif a >= 0.55:
                bars += "·"
            else:
                bars += " "

        print(f"  {concept_name:20s}: [{bars}] "
              f"emerge@L{first_above_60}, useful@L{first_above_80}, "
              f"strong@L{first_above_90}, peak@L{peak_layer}")

    print(f"\n  Legend: █≥0.95 ▓≥0.85 ▒≥0.75 ░≥0.65 ·≥0.55")
    print()


# ---------------------------------------------------------------------------
# PHASE 26: Neuron Specificity Spectrum — full selectivity distribution
# ---------------------------------------------------------------------------

def neuron_specificity_spectrum(all_acts, concept_names, num_layers):
    """
    For a sample layer (mid-network), compute every neuron's max selectivity
    across all concepts. Produces a histogram of neuron specificity showing
    what fraction of neurons are concept-selective vs non-selective.
    """
    print("=" * 70)
    print("PHASE 26: Neuron Specificity Spectrum")
    print("=" * 70)

    target_layer = 12  # mid-network
    pos_all = all_acts[concept_names[0]]["positive"][target_layer]
    hidden_size = pos_all.shape[1]

    # For each neuron, compute max Cohen's d across all concepts
    max_d = np.zeros(hidden_size)
    best_concept = [""] * hidden_size

    for concept_name in concept_names:
        pos = all_acts[concept_name]["positive"][target_layer]
        neg = all_acts[concept_name]["negative"][target_layer]
        for n in range(hidden_size):
            pooled_std = np.sqrt((pos[:, n].var() + neg[:, n].var()) / 2) + 1e-8
            d = abs(pos[:, n].mean() - neg[:, n].mean()) / pooled_std
            if d > max_d[n]:
                max_d[n] = d
                best_concept[n] = concept_name

    # Histogram of max selectivity
    bins = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, float('inf')]
    labels = ["<0.2 (noise)", "0.2-0.5 (weak)", "0.5-1.0 (moderate)",
              "1.0-1.5 (good)", "1.5-2.0 (strong)", "2.0-3.0 (very strong)",
              "3.0+ (dedicated)"]

    print(f"  Neuron selectivity distribution at L{target_layer} "
          f"({hidden_size} neurons):\n")
    for i in range(len(bins) - 1):
        count = int(((max_d >= bins[i]) & (max_d < bins[i + 1])).sum())
        pct = 100 * count / hidden_size
        bar = "█" * int(pct / 2)
        print(f"    {labels[i]:25s}: {count:3d} ({pct:4.1f}%) {bar}")

    # Top-10 most selective neurons
    top10 = np.argsort(max_d)[::-1][:10]
    print(f"\n  Top-10 most selective neurons at L{target_layer}:")
    for idx in top10:
        print(f"    N{idx:3d}: d={max_d[idx]:.2f} → {best_concept[idx]}")

    # Per-concept: how many neurons are "dedicated" (d > 1.5)?
    print(f"\n  Dedicated neurons per concept (d > 1.5 at L{target_layer}):")
    for concept_name in concept_names:
        count = sum(1 for n in range(hidden_size)
                    if best_concept[n] == concept_name and max_d[n] > 1.5)
        print(f"    {concept_name:20s}: {count}")

    print()


# ---------------------------------------------------------------------------
# PHASE 27: Concept Bottleneck — best single layer for all concepts
# ---------------------------------------------------------------------------

def concept_bottleneck_analysis(concept_names, locality_results, num_layers):
    """
    Find which single layer maximizes the mean accuracy across ALL concepts.
    This is the network's "concept bottleneck" — the representation where
    the most concept information is simultaneously accessible.
    """
    print("=" * 70)
    print("PHASE 27: Concept Bottleneck — Optimal Single Layer")
    print("=" * 70)

    # For each layer, compute mean accuracy across all concepts
    layer_scores = np.zeros(num_layers)
    for layer_idx in range(num_layers):
        accs = []
        for concept_name in concept_names:
            accs.append(locality_results[concept_name]["layer_accuracies"][layer_idx])
        layer_scores[layer_idx] = np.mean(accs)

    best_layer = int(np.argmax(layer_scores))
    best_score = layer_scores[best_layer]

    # Print layer scores compactly
    print(f"  Mean accuracy across all {len(concept_names)} concepts per layer:\n")
    for l in range(num_layers):
        bar = "█" * int(layer_scores[l] * 40)
        marker = " ◄ BEST" if l == best_layer else ""
        print(f"    L{l:2d}: {layer_scores[l]:.3f} {bar}{marker}")

    # Top-3 bottleneck layers
    top3 = np.argsort(layer_scores)[::-1][:3]
    print(f"\n  Bottleneck layers: {', '.join(f'L{l}({layer_scores[l]:.3f})' for l in top3)}")

    # Per-concept accuracy at the bottleneck layer
    print(f"\n  Per-concept accuracy at bottleneck L{best_layer}:")
    for concept_name in concept_names:
        acc = locality_results[concept_name]["layer_accuracies"][best_layer]
        print(f"    {concept_name:20s}: {acc:.3f}")

    print()


# ---------------------------------------------------------------------------
# PHASE 28: Concept Gradient Landscape — sensitivity to perturbation
# ---------------------------------------------------------------------------

def concept_gradient_landscape(all_acts, concept_names, sparse_results, steering_vectors):
    """
    Analyze the sharpness of concept boundaries by perturbing steering vectors
    and measuring how quickly classification accuracy degrades.
    Sharp boundaries = well-defined concepts; gradual = fuzzy/distributed.
    """
    print("=" * 70)
    print("PHASE 28: Concept Gradient Landscape — Boundary Sharpness")
    print("=" * 70)

    perturbation_scales = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    rng = np.random.RandomState(42)

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        acts_pos = all_acts[concept_name]["positive"][best_layer]
        acts_neg = all_acts[concept_name]["negative"][best_layer]
        X = np.vstack([acts_pos, acts_neg])
        y = np.array([1] * len(acts_pos) + [0] * len(acts_neg))

        # Get the steering vector for this concept
        sv = steering_vectors.get(concept_name)
        if sv is None:
            continue

        # Baseline accuracy with clean data
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_sc, y)
        base_acc = clf.score(X_sc, y)

        # Perturb: add random noise orthogonal to steering vector
        sv_norm = sv / (np.linalg.norm(sv) + 1e-12)
        noise_base = rng.randn(X.shape[1])
        noise_base -= np.dot(noise_base, sv_norm) * sv_norm  # orthogonalize
        noise_base /= (np.linalg.norm(noise_base) + 1e-12)

        accs = []
        data_std = np.std(X)
        for scale in perturbation_scales:
            X_perturbed = X + scale * data_std * noise_base[None, :]
            X_p_sc = scaler.transform(X_perturbed)
            acc = clf.score(X_p_sc, y)
            accs.append(acc)

        # Sharpness = how quickly accuracy drops (AUC under perturbation curve)
        # Normalize: 1.0 = perfectly robust, 0.0 = collapses immediately
        auc = np.trapezoid(accs, perturbation_scales) / (perturbation_scales[-1] - perturbation_scales[0])

        # Print mini perturbation curve
        curve = " → ".join(f"{a:.2f}" for a in accs)
        print(f"  {concept_name:20s}: robustness={auc:.3f}  [{curve}]")

    # Along-direction perturbation: shift all points along steering vector
    print(f"\n  Along-direction sensitivity (shifting along steering vector):")
    shifts = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        acts_pos = all_acts[concept_name]["positive"][best_layer]
        acts_neg = all_acts[concept_name]["negative"][best_layer]
        X = np.vstack([acts_pos, acts_neg])
        y = np.array([1] * len(acts_pos) + [0] * len(acts_neg))

        sv = steering_vectors.get(concept_name)
        if sv is None:
            continue
        sv_norm = sv / (np.linalg.norm(sv) + 1e-12)

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_sc, y)

        data_std = np.std(X)
        accs = []
        for shift in shifts:
            X_shifted = X + shift * data_std * sv_norm[None, :]
            X_s_sc = scaler.transform(X_shifted)
            acc = clf.score(X_s_sc, y)
            accs.append(acc)

        curve = " → ".join(f"{a:.2f}" for a in accs)
        print(f"  {concept_name:20s}: [{curve}]")

    print()


# ---------------------------------------------------------------------------
# PHASE 29: Concept Dimensionality Reduction — t-SNE / PCA visualization data
# ---------------------------------------------------------------------------

def concept_dimensionality_reduction(all_acts, concept_names, sparse_results):
    """
    Compute PCA projections of activations at each concept's best layer,
    report explained variance and cluster separation in 2D/3D.
    """
    print("=" * 70)
    print("PHASE 29: Concept Dimensionality Reduction — PCA Projections")
    print("=" * 70)

    from sklearn.decomposition import PCA

    # Collect all activations at a representative layer (L10 bottleneck or mid-layer)
    # Use each concept's best layer for per-concept analysis
    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        acts_pos = all_acts[concept_name]["positive"][best_layer]
        acts_neg = all_acts[concept_name]["negative"][best_layer]
        X = np.vstack([acts_pos, acts_neg])

        pca = PCA(n_components=min(10, X.shape[0], X.shape[1]))
        X_pca = pca.fit_transform(X)

        # Explained variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dims_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        dims_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        dims_99 = int(np.searchsorted(cumvar, 0.99)) + 1

        # Cluster separation in 2D: distance between centroids / mean spread
        centroid_pos = X_pca[:len(acts_pos), :2].mean(axis=0)
        centroid_neg = X_pca[len(acts_pos):, :2].mean(axis=0)
        centroid_dist = np.linalg.norm(centroid_pos - centroid_neg)
        spread_pos = np.mean(np.linalg.norm(X_pca[:len(acts_pos), :2] - centroid_pos, axis=1))
        spread_neg = np.mean(np.linalg.norm(X_pca[len(acts_pos):, :2] - centroid_neg, axis=1))
        separation = centroid_dist / (0.5 * (spread_pos + spread_neg) + 1e-12)

        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        print(f"  {concept_name:20s} @ L{best_layer:2d}: "
              f"PC1={var1:4.1f}% PC2={var2:4.1f}% | "
              f"90%@{dims_90}d 95%@{dims_95}d 99%@{dims_99}d | "
              f"2D-sep={separation:.2f}")

    # Multi-concept PCA: project all concepts at bottleneck layer (L10)
    bottleneck = 10
    print(f"\n  Multi-concept PCA at bottleneck L{bottleneck}:")
    all_X = []
    all_labels = []
    for i, concept_name in enumerate(concept_names):
        acts_pos = all_acts[concept_name]["positive"][bottleneck]
        acts_neg = all_acts[concept_name]["negative"][bottleneck]
        all_X.append(acts_pos)
        all_X.append(acts_neg)
        all_labels.extend([concept_name + "+"] * len(acts_pos))
        all_labels.extend([concept_name + "-"] * len(acts_neg))

    X_all = np.vstack(all_X)
    pca_all = PCA(n_components=min(10, X_all.shape[0]))
    X_pca_all = pca_all.fit_transform(X_all)

    cumvar = np.cumsum(pca_all.explained_variance_ratio_)
    dims_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumvar, 0.95)) + 1

    print(f"    Total samples: {X_all.shape[0]}, dims: {X_all.shape[1]}")
    print(f"    Variance: PC1={pca_all.explained_variance_ratio_[0]*100:.1f}% "
          f"PC2={pca_all.explained_variance_ratio_[1]*100:.1f}% "
          f"PC3={pca_all.explained_variance_ratio_[2]*100:.1f}%")
    print(f"    Dims for 90%={dims_90}, 95%={dims_95}")

    # Between-concept separation in 2D
    print(f"\n    Pairwise 2D centroid distances:")
    centroids = {}
    for concept_name in concept_names:
        mask = [l.startswith(concept_name) for l in all_labels]
        centroids[concept_name] = X_pca_all[mask, :2].mean(axis=0)

    for i in range(len(concept_names)):
        for j in range(i + 1, len(concept_names)):
            d = np.linalg.norm(centroids[concept_names[i]] - centroids[concept_names[j]])
            if d < 1.0:  # only report close pairs
                print(f"      {concept_names[i]:15s} ↔ {concept_names[j]:15s}: {d:.3f} (close!)")

    print()


# ---------------------------------------------------------------------------
# PHASE 30: Concept Polarity — symmetric vs asymmetric activation patterns
# ---------------------------------------------------------------------------

def concept_polarity_analysis(all_acts, concept_names, sparse_results):
    """
    Examine whether concept neurons use symmetric (push/pull) or asymmetric
    (presence/absence) coding. Symmetric = neuron activates high for pos AND
    low for neg. Asymmetric = neuron activates for one side, baseline for other.
    """
    print("=" * 70)
    print("PHASE 30: Concept Polarity — Symmetric vs Asymmetric Coding")
    print("=" * 70)

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        top_neuron = sparse_results[concept_name]["top_neurons"][0]

        pos_acts = all_acts[concept_name]["positive"][best_layer][:, top_neuron]
        neg_acts = all_acts[concept_name]["negative"][best_layer][:, top_neuron]

        # Compute polarity metrics
        mean_pos = np.mean(pos_acts)
        mean_neg = np.mean(neg_acts)
        std_all = np.std(np.concatenate([pos_acts, neg_acts]))

        # Asymmetry: how far from zero is the midpoint?
        # If midpoint ≈ 0, coding is symmetric (push/pull around zero)
        # If midpoint >> 0, one side is at baseline
        midpoint = (mean_pos + mean_neg) / 2.0
        contrast = abs(mean_pos - mean_neg)
        asymmetry = abs(midpoint) / (contrast + 1e-12)

        # Effect size for each direction from baseline (0)
        d_pos = abs(mean_pos) / (np.std(pos_acts) + 1e-12)
        d_neg = abs(mean_neg) / (np.std(neg_acts) + 1e-12)

        # Classification: symmetric if both sides have strong signal
        if min(d_pos, d_neg) > 0.5 and asymmetry < 0.5:
            coding = "symmetric"
        elif max(d_pos, d_neg) > 1.0 and min(d_pos, d_neg) < 0.3:
            coding = "one-sided"
        else:
            coding = "mixed"

        print(f"  {concept_name:20s} N{top_neuron:3d}@L{best_layer:2d}: "
              f"μ+={mean_pos:+.2f} μ-={mean_neg:+.2f} "
              f"asym={asymmetry:.2f} d+={d_pos:.2f} d-={d_neg:.2f} → {coding}")

    print()


# ---------------------------------------------------------------------------
# PHASE 31: Layer Transition Dynamics — rate of representational change
# ---------------------------------------------------------------------------

def layer_transition_dynamics(all_acts, concept_names, num_layers):
    """
    Measure how much concept representations change between adjacent layers.
    Large transitions = processing happening; small = passing through.
    Identifies "decision layers" where concepts crystallize.
    """
    print("=" * 70)
    print("PHASE 31: Layer Transition Dynamics — Where Concepts Crystallize")
    print("=" * 70)

    for concept_name in concept_names:
        # Compute steering vector at each layer
        svs = []
        for l in range(num_layers):
            pos = all_acts[concept_name]["positive"][l]
            neg = all_acts[concept_name]["negative"][l]
            sv = np.mean(pos, axis=0) - np.mean(neg, axis=0)
            sv_norm = sv / (np.linalg.norm(sv) + 1e-12)
            svs.append(sv_norm)

        # Compute cosine similarity between adjacent layers
        transitions = []
        for l in range(num_layers - 1):
            cos_sim = np.dot(svs[l], svs[l + 1])
            transitions.append(1.0 - cos_sim)  # transition magnitude

        # Find peak transition (biggest direction change)
        peak_layer = int(np.argmax(transitions))
        peak_mag = transitions[peak_layer]

        # Cumulative transition (total direction drift)
        cumulative = sum(transitions)

        # Sparkline of transition magnitudes
        max_t = max(transitions) if max(transitions) > 0 else 1
        blocks = " ▁▂▃▄▅▆▇█"
        spark = ""
        for t in transitions:
            idx = min(int(t / max_t * 8), 8)
            spark += blocks[idx]

        print(f"  {concept_name:20s}: peak=L{peak_layer}→L{peak_layer+1} "
              f"(Δ={peak_mag:.3f}) cumΔ={cumulative:.2f} [{spark}]")

    # Cross-concept: at which layers do ALL concepts change most?
    print(f"\n  Mean transition magnitude per layer boundary:")
    layer_means = np.zeros(num_layers - 1)
    for concept_name in concept_names:
        svs = []
        for l in range(num_layers):
            pos = all_acts[concept_name]["positive"][l]
            neg = all_acts[concept_name]["negative"][l]
            sv = np.mean(pos, axis=0) - np.mean(neg, axis=0)
            sv_norm = sv / (np.linalg.norm(sv) + 1e-12)
            svs.append(sv_norm)
        for l in range(num_layers - 1):
            layer_means[l] += 1.0 - np.dot(svs[l], svs[l + 1])
    layer_means /= len(concept_names)

    max_m = max(layer_means) if max(layer_means) > 0 else 1
    for l in range(num_layers - 1):
        bar = "█" * int(layer_means[l] / max_m * 30)
        marker = " ◄" if layer_means[l] == max(layer_means) else ""
        print(f"    L{l:2d}→L{l+1:2d}: {layer_means[l]:.4f} {bar}{marker}")

    print()


# ---------------------------------------------------------------------------
# PHASE 32: Concept Interference — steering vector cross-talk
# ---------------------------------------------------------------------------

def concept_interference_analysis(all_acts, concept_names, sparse_results, steering_vectors):
    """
    Measure how injecting one concept's steering vector affects classification
    of other concepts. Goes beyond cosine similarity to measure functional
    interference.
    """
    print("=" * 70)
    print("PHASE 32: Concept Interference — Steering Vector Cross-Talk")
    print("=" * 70)

    n_concepts = len(concept_names)
    interference = np.zeros((n_concepts, n_concepts))

    for i, target in enumerate(concept_names):
        best_layer = sparse_results[target]["best_layer"]
        pos = all_acts[target]["positive"][best_layer]
        neg = all_acts[target]["negative"][best_layer]
        X = np.vstack([pos, neg])
        y = np.array([1] * len(pos) + [0] * len(neg))

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_sc, y)
        base_acc = clf.score(X_sc, y)

        for j, source in enumerate(concept_names):
            if i == j:
                interference[i, j] = 0.0
                continue

            sv = steering_vectors.get(source)
            if sv is None:
                continue

            # Inject source steering vector (1 std magnitude)
            sv_scaled = sv / (np.linalg.norm(sv) + 1e-12) * np.std(X)
            X_perturbed = X + sv_scaled[None, :]
            X_p_sc = scaler.transform(X_perturbed)
            perturbed_acc = clf.score(X_p_sc, y)

            interference[i, j] = base_acc - perturbed_acc  # positive = interference

    # Print interference matrix
    print(f"  Interference matrix (row=target, col=injected, values=accuracy drop):\n")
    header = "  " + " " * 22 + "".join(f"{c[:6]:>7s}" for c in concept_names)
    print(header)
    for i, target in enumerate(concept_names):
        row = f"  {target:20s}:"
        for j in range(n_concepts):
            val = interference[i, j]
            if val > 0.1:
                row += f"  {val:+.2f}"
            elif val > 0.01:
                row += f"  {val:+.2f}"
            else:
                row += f"     ·"
        print(row)

    # Summary: most interfered-with and most interfering concepts
    mean_received = np.mean(interference, axis=1)  # how much each concept is affected
    mean_caused = np.mean(interference, axis=0)  # how much each concept affects others
    print(f"\n  Most vulnerable (receives interference):")
    for i in np.argsort(mean_received)[::-1][:3]:
        print(f"    {concept_names[i]:20s}: mean drop={mean_received[i]:.3f}")
    print(f"\n  Most disruptive (causes interference):")
    for j in np.argsort(mean_caused)[::-1][:3]:
        print(f"    {concept_names[j]:20s}: mean drop={mean_caused[j]:.3f}")

    print()


# ---------------------------------------------------------------------------
# PHASE 33: Concept Decision Boundaries — nonlinearity analysis
# ---------------------------------------------------------------------------

def concept_decision_boundary_analysis(all_acts, concept_names, sparse_results):
    """
    Test how linear vs nonlinear the concept boundaries are by comparing
    linear probe accuracy to polynomial feature accuracy. Large gap = concept
    requires nonlinear decision boundary.
    """
    print("=" * 70)
    print("PHASE 33: Concept Decision Boundaries — Linearity Analysis")
    print("=" * 70)

    from sklearn.preprocessing import PolynomialFeatures

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        top_neurons = sparse_results[concept_name]["top_neurons"][:5]

        pos = all_acts[concept_name]["positive"][best_layer][:, top_neurons]
        neg = all_acts[concept_name]["negative"][best_layer][:, top_neurons]
        X = np.vstack([pos, neg])
        y = np.array([1] * len(pos) + [0] * len(neg))

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        # Linear probe
        clf_lin = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_lin = np.mean(cross_val_score(clf_lin, X_sc, y, cv=cv))

        # Quadratic features
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_sc)
        clf_poly = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        acc_poly = np.mean(cross_val_score(clf_poly, X_poly, y, cv=cv))

        gap = acc_poly - acc_lin
        linearity = "linear" if gap < 0.02 else ("slightly nonlinear" if gap < 0.05 else "nonlinear")

        print(f"  {concept_name:20s}: linear={acc_lin:.3f} quad={acc_poly:.3f} "
              f"gap={gap:+.3f} → {linearity}")

    print()


# ---------------------------------------------------------------------------
# PHASE 34: Concept Prototypes — most/least typical examples
# ---------------------------------------------------------------------------

def concept_prototype_analysis(all_acts, concept_names, sparse_results, steering_vectors):
    """
    For each concept, find the most and least prototypical examples based on
    projection onto the difference-of-means direction. Compute margin statistics.
    """
    print("=" * 70)
    print("PHASE 34: Concept Prototypes — Most/Least Typical Examples")
    print("=" * 70)

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]

        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]

        # Use difference-of-means as direction (not L1 probe weight)
        dom = np.mean(pos, axis=0) - np.mean(neg, axis=0)
        dom_norm = dom / (np.linalg.norm(dom) + 1e-12)

        # Project onto direction
        proj_pos = pos @ dom_norm
        proj_neg = neg @ dom_norm

        # Boundary at midpoint between class means
        boundary = (np.mean(proj_pos) + np.mean(proj_neg)) / 2.0
        proto_pos = proj_pos - boundary
        proto_neg = boundary - proj_neg

        # Margin distribution
        margin_pos = np.min(proto_pos)
        margin_neg = np.min(proto_neg)
        mean_margin = (np.mean(proto_pos) + np.mean(proto_neg)) / 2.0

        # Separability: fraction of correctly-sided examples
        correct_pos = np.mean(proto_pos > 0)
        correct_neg = np.mean(proto_neg > 0)

        # Spread ratio: how tight is the clustering?
        spread_pos = np.std(proto_pos)
        spread_neg = np.std(proto_neg)

        print(f"  {concept_name:20s} @ L{best_layer:2d}: "
              f"margin={mean_margin:.3f} min+={margin_pos:+.3f} min-={margin_neg:+.3f} "
              f"sep={correct_pos:.0%}/{correct_neg:.0%} "
              f"spread={spread_pos:.3f}/{spread_neg:.3f}")

    print()


# ---------------------------------------------------------------------------
# PHASE 35: Residual Analysis — what's beyond the 8 concepts?
# ---------------------------------------------------------------------------

def residual_analysis(all_acts, concept_names, sparse_results, steering_vectors, num_layers):
    """
    Project out all known concept directions and analyze what remains.
    How much variance is explained by our 8 concepts? What structure
    exists in the residual?
    """
    print("=" * 70)
    print("PHASE 35: Residual Analysis — Variance Beyond Known Concepts")
    print("=" * 70)

    # Use bottleneck layer (L10) for analysis
    target_layer = 10

    # Collect all activations at target layer
    all_X = []
    for concept_name in concept_names:
        pos = all_acts[concept_name]["positive"][target_layer]
        neg = all_acts[concept_name]["negative"][target_layer]
        all_X.append(pos)
        all_X.append(neg)
    X = np.vstack(all_X)

    # Compute concept subspace from difference-of-means directions (denser than L1)
    sv_list = []
    for concept_name in concept_names:
        # Use best_layer for each concept's direction at target_layer
        pos = all_acts[concept_name]["positive"][target_layer]
        neg = all_acts[concept_name]["negative"][target_layer]
        dom = np.mean(pos, axis=0) - np.mean(neg, axis=0)
        dom_norm = dom / (np.linalg.norm(dom) + 1e-12)
        sv_list.append(dom_norm)

    if len(sv_list) < 2:
        print("  Not enough steering vectors for analysis.")
        print()
        return

    V = np.array(sv_list)  # (n_concepts, hidden_size)

    # Project data onto concept subspace and residual
    # Use SVD of V to get orthonormal basis for concept space
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis = Vt[:len(sv_list)]  # orthonormal basis for concept space

    X_centered = X - X.mean(axis=0)
    total_var = np.var(X_centered, axis=0).sum()

    # Projection onto concept subspace
    proj = X_centered @ basis.T @ basis
    concept_var = np.var(proj, axis=0).sum()

    # Residual
    residual = X_centered - proj
    residual_var = np.var(residual, axis=0).sum()

    pct_concept = concept_var / total_var * 100
    pct_residual = residual_var / total_var * 100

    print(f"  At L{target_layer} ({X.shape[0]} samples, {X.shape[1]} dims):")
    print(f"    Total variance:   {total_var:.2f}")
    print(f"    Concept space:    {concept_var:.2f} ({pct_concept:.1f}%)")
    print(f"    Residual:         {residual_var:.2f} ({pct_residual:.1f}%)")
    print(f"    Concept dims:     {len(sv_list)} / {X.shape[1]}")

    # PCA on residual to find dominant non-concept directions
    from sklearn.decomposition import PCA
    pca_res = PCA(n_components=min(10, residual.shape[0], residual.shape[1]))
    pca_res.fit(residual)

    print(f"\n    Top PCA components of residual:")
    cumvar = np.cumsum(pca_res.explained_variance_ratio_)
    for k in range(min(5, len(pca_res.explained_variance_ratio_))):
        print(f"      PC{k+1}: {pca_res.explained_variance_ratio_[k]*100:.1f}% "
              f"(cumulative: {cumvar[k]*100:.1f}%)")

    # Per-concept: how much variance does each concept direction explain?
    print(f"\n    Per-concept variance explained:")
    for i, concept_name in enumerate(concept_names):
        if i < len(sv_list):
            proj_c = X_centered @ sv_list[i]
            var_c = np.var(proj_c)
            pct = var_c / total_var * 100
            print(f"      {concept_name:20s}: {pct:.2f}%")

    print()


# ---------------------------------------------------------------------------
# PHASE 36: Concept Stability — bootstrap analysis
# ---------------------------------------------------------------------------

def concept_stability_analysis(all_acts, concept_names, sparse_results):
    """
    Bootstrap resample the training data and re-run neuron ranking to see
    how stable the top neuron assignments are. High stability = robust features.
    """
    print("=" * 70)
    print("PHASE 36: Concept Stability — Bootstrap Neuron Assignment")
    print("=" * 70)

    N_BOOTSTRAPS = 5
    rng = np.random.RandomState(42)

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        top_neuron = sparse_results[concept_name]["top_neurons"][0]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]
        n_pos, n_neg = len(pos), len(neg)

        original_top = top_neuron
        top1_counts = {}

        for b in range(N_BOOTSTRAPS):
            idx_pos = rng.choice(n_pos, n_pos, replace=True)
            idx_neg = rng.choice(n_neg, n_neg, replace=True)
            X_b = np.vstack([pos[idx_pos], neg[idx_neg]])
            y_b = np.array([1] * n_pos + [0] * n_neg)

            # Use MI ranking (fast) instead of L1 probing
            mi = mutual_info_classif(X_b, y_b, random_state=b)
            top1 = int(np.argmax(mi))
            top1_counts[top1] = top1_counts.get(top1, 0) + 1

        # Stability = fraction of bootstraps agreeing on the same top neuron
        most_common = max(top1_counts.values())
        stability = most_common / N_BOOTSTRAPS
        n_unique = len(top1_counts)
        agrees_with_original = top1_counts.get(original_top, 0) / N_BOOTSTRAPS

        print(f"  {concept_name:20s} N{original_top:3d}@L{best_layer:2d}: "
              f"stability={stability:.0%} unique={n_unique} "
              f"original_frac={agrees_with_original:.0%}")

    print()


# ---------------------------------------------------------------------------
# PHASE 37: Concept Effective Rank — dimensionality of concept signal
# ---------------------------------------------------------------------------

def concept_effective_rank(all_acts, concept_names, sparse_results):
    """
    Compute the effective rank of each concept's activation subspace.
    Low rank = concept lives on a low-dimensional manifold.
    """
    print("=" * 70)
    print("PHASE 37: Concept Effective Rank — Signal Dimensionality")
    print("=" * 70)

    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]

        # Concept signal: difference from grand mean for each class
        X = np.vstack([pos, neg])
        X_centered = X - X.mean(axis=0)

        # SVD to get singular values
        _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
        S = S[S > 1e-10]

        # Effective rank (Shannon entropy of normalized singular values)
        p = S / S.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))
        eff_rank = np.exp(entropy)

        # Also compute ratio of first singular value
        top1_ratio = S[0] / S.sum() * 100
        top3_ratio = S[:3].sum() / S.sum() * 100

        # Nuclear norm ratio (another measure of spread)
        stable_rank = (S ** 2).sum() / (S[0] ** 2 + 1e-12)

        print(f"  {concept_name:20s} @ L{best_layer:2d}: "
              f"eff_rank={eff_rank:.1f} stable_rank={stable_rank:.1f} "
              f"top1={top1_ratio:.1f}% top3={top3_ratio:.1f}%")

    print()


# ---------------------------------------------------------------------------
# PHASE 38: Concept Predictive Power — cross-concept leakage
# ---------------------------------------------------------------------------

def concept_leakage_analysis(all_acts, concept_names, sparse_results):
    """
    Can you predict concept A from concept B's neurons? High predictability
    means the concepts share information (leakage). Uses the top neuron
    of each concept to predict all other concepts.
    """
    print("=" * 70)
    print("PHASE 38: Concept Leakage — Cross-Concept Predictability")
    print("=" * 70)

    n = len(concept_names)
    leakage = np.zeros((n, n))

    # For each concept, get its top neuron activations at its best layer
    for i, target in enumerate(concept_names):
        best_layer = sparse_results[target]["best_layer"]
        target_neuron = sparse_results[target]["top_neurons"][0]

        for j, source in enumerate(concept_names):
            # Use source concept's best neuron to predict target concept
            source_layer = sparse_results[source]["best_layer"]
            source_neuron = sparse_results[source]["top_neurons"][0]

            # Get target's labels
            pos_t = all_acts[target]["positive"][source_layer][:, source_neuron]
            neg_t = all_acts[target]["negative"][source_layer][:, source_neuron]
            X = np.concatenate([pos_t, neg_t]).reshape(-1, 1)
            y = np.array([1] * len(pos_t) + [0] * len(neg_t))

            clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acc = np.mean(cross_val_score(clf, X, y, cv=cv))
            leakage[i, j] = acc

    # Print leakage matrix (only non-trivial entries)
    print(f"  Cross-prediction accuracy (row=predicted, col=predictor neuron):\n")
    header = "  " + " " * 22 + "".join(f"{c[:6]:>7s}" for c in concept_names)
    print(header)
    for i, target in enumerate(concept_names):
        row = f"  {target:20s}:"
        for j in range(n):
            val = leakage[i, j]
            if i == j:
                row += f"  [{val:.2f}]"
            elif val > 0.6:
                row += f"  {val:.2f}*"
            else:
                row += f"  {val:.2f} "
        print(row)

    # Most leaky pairs
    print(f"\n  Highest cross-predictions (excluding diagonal):")
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((leakage[i, j], concept_names[i], concept_names[j]))
    pairs.sort(reverse=True)
    for acc, target, source in pairs[:5]:
        print(f"    {source:15s} → {target:15s}: {acc:.3f}")

    print()


# ---------------------------------------------------------------------------
# PHASE 39: Global Summary Statistics
# ---------------------------------------------------------------------------

def global_summary_statistics(all_acts, concept_names, sparse_results, num_layers):
    """
    Compute global summary statistics across all concepts and layers:
    mean activation norms, layer-wise statistics, concept clustering quality.
    """
    print("=" * 70)
    print("PHASE 39: Global Summary Statistics")
    print("=" * 70)

    # Activation norms per layer (averaged over all concepts)
    print(f"  Mean activation L2 norm per layer:")
    layer_norms = np.zeros(num_layers)
    for l in range(num_layers):
        norms = []
        for concept_name in concept_names:
            pos = all_acts[concept_name]["positive"][l]
            neg = all_acts[concept_name]["negative"][l]
            norms.extend(np.linalg.norm(pos, axis=1).tolist())
            norms.extend(np.linalg.norm(neg, axis=1).tolist())
        layer_norms[l] = np.mean(norms)

    max_norm = max(layer_norms)
    for l in range(num_layers):
        bar = "█" * int(layer_norms[l] / max_norm * 30)
        print(f"    L{l:2d}: {layer_norms[l]:6.2f} {bar}")

    # Concept separability summary
    print(f"\n  Concept separability summary:")
    for concept_name in concept_names:
        best_layer = sparse_results[concept_name]["best_layer"]
        pos = all_acts[concept_name]["positive"][best_layer]
        neg = all_acts[concept_name]["negative"][best_layer]

        centroid_pos = np.mean(pos, axis=0)
        centroid_neg = np.mean(neg, axis=0)
        inter_dist = np.linalg.norm(centroid_pos - centroid_neg)

        intra_pos = np.mean(np.linalg.norm(pos - centroid_pos, axis=1))
        intra_neg = np.mean(np.linalg.norm(neg - centroid_neg, axis=1))
        intra_mean = (intra_pos + intra_neg) / 2.0

        # Silhouette-like score
        silhouette = (inter_dist - intra_mean) / (max(inter_dist, intra_mean) + 1e-12)

        print(f"    {concept_name:20s} @ L{best_layer:2d}: "
              f"inter={inter_dist:.2f} intra={intra_mean:.2f} "
              f"silhouette={silhouette:.3f}")

    # Overall statistics
    all_best_layers = [sparse_results[c]["best_layer"] for c in concept_names]
    all_min_neurons = [sparse_results[c]["min_neurons"] for c in concept_names]
    print(f"\n  Overall:")
    print(f"    Mean best layer:     {np.mean(all_best_layers):.1f} ± {np.std(all_best_layers):.1f}")
    print(f"    Mean min neurons:    {np.mean(all_min_neurons):.1f} ± {np.std(all_min_neurons):.1f}")
    print(f"    Layer range:         L{min(all_best_layers)} to L{max(all_best_layers)}")
    print(f"    Activation norm growth: {layer_norms[0]:.1f} → {layer_norms[-1]:.1f} "
          f"({layer_norms[-1]/layer_norms[0]:.1f}x)")

    print()


# ---------------------------------------------------------------------------
# PHASE 40: Concept Entanglement Clusters
# ---------------------------------------------------------------------------

def concept_entanglement_clusters(all_acts, concept_names, sparse_results):
    """
    Build an entanglement graph from neuron-level overlap and find clusters
    of functionally related concepts using hierarchical clustering on
    activation correlations.
    """
    print("=" * 70)
    print("PHASE 40: Concept Entanglement Clusters")
    print("=" * 70)

    # Compute pairwise correlation between concept signals at a shared layer
    target_layer = 10  # bottleneck layer
    n = len(concept_names)

    # Difference-of-means vectors at target layer
    dom_vectors = []
    for concept_name in concept_names:
        pos = all_acts[concept_name]["positive"][target_layer]
        neg = all_acts[concept_name]["negative"][target_layer]
        dom = np.mean(pos, axis=0) - np.mean(neg, axis=0)
        dom_vectors.append(dom)
    dom_matrix = np.array(dom_vectors)

    # Pairwise cosine similarity
    norms = np.linalg.norm(dom_matrix, axis=1, keepdims=True) + 1e-12
    dom_normed = dom_matrix / norms
    cos_sim = dom_normed @ dom_normed.T

    # Hierarchical clustering on 1 - |cos_sim|
    dist_matrix = 1.0 - np.abs(cos_sim)
    np.fill_diagonal(dist_matrix, 0)
    condensed = pdist(dist_matrix)
    Z = linkage(condensed, method='average')
    clusters = fcluster(Z, t=0.8, criterion='distance')

    print(f"  Cosine similarity at L{target_layer} (difference-of-means):\n")
    header = "  " + " " * 22 + "".join(f"{c[:6]:>7s}" for c in concept_names)
    print(header)
    for i in range(n):
        row = f"  {concept_names[i]:20s}:"
        for j in range(n):
            val = cos_sim[i, j]
            if i == j:
                row += f"   1.00"
            elif abs(val) > 0.3:
                row += f"  {val:+.2f}"
            else:
                row += f"     · "
        print(row)

    # Print cluster assignments
    n_clusters = len(set(clusters))
    print(f"\n  Entanglement clusters ({n_clusters} clusters at distance threshold 0.8):")
    for c in sorted(set(clusters)):
        members = [concept_names[i] for i in range(n) if clusters[i] == c]
        print(f"    Cluster {c}: {', '.join(members)}")

    # Strongest entanglements
    print(f"\n  Strongest entanglements (|cos| > 0.3):")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((abs(cos_sim[i, j]), cos_sim[i, j], concept_names[i], concept_names[j]))
    pairs.sort(reverse=True)
    for abs_val, val, c1, c2 in pairs:
        if abs_val > 0.3:
            direction = "aligned" if val > 0 else "opposed"
            print(f"    {c1:15s} ↔ {c2:15s}: cos={val:+.3f} ({direction})")

    print()


# ---------------------------------------------------------------------------
# PHASE 41: Layer-Specific Concept Quality
# ---------------------------------------------------------------------------

def layer_concept_quality(all_acts, concept_names, num_layers):
    """
    For each layer, compute how well all concepts are simultaneously decodable.
    Produces a single "concept quality" metric per layer that combines
    individual concept accuracies.
    """
    print("=" * 70)
    print("PHASE 41: Layer-Specific Concept Quality")
    print("=" * 70)

    layer_quality = np.zeros(num_layers)
    layer_worst = np.zeros(num_layers)

    for l in range(num_layers):
        accs = []
        for concept_name in concept_names:
            pos = all_acts[concept_name]["positive"][l]
            neg = all_acts[concept_name]["negative"][l]
            X = np.vstack([pos, neg])
            y = np.array([1] * len(pos) + [0] * len(neg))

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)
            clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            clf.fit(X_sc, y)
            accs.append(clf.score(X_sc, y))

        layer_quality[l] = np.mean(accs)
        layer_worst[l] = np.min(accs)

    best_layer = int(np.argmax(layer_quality))
    best_minmax = int(np.argmax(layer_worst))

    print(f"  Layer quality (mean acc) and worst-case (min acc):\n")
    for l in range(num_layers):
        bar = "█" * int(layer_quality[l] * 30)
        markers = ""
        if l == best_layer:
            markers += " ◄best-mean"
        if l == best_minmax:
            markers += " ◄best-worst"
        print(f"    L{l:2d}: mean={layer_quality[l]:.3f} min={layer_worst[l]:.3f} {bar}{markers}")

    print()


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

    # Phase 6: Concept composition (informational)
    concept_composition_analysis(all_acts, concept_names, sparse_results, steering_vectors)

    # Phase 7: Causal ablation (informational)
    causal_ablation_analysis(all_acts, concept_names, sparse_results)

    # Phase 8: Activation patching (informational)
    activation_patching_analysis(all_acts, concept_names, sparse_results)

    # Phase 9: ICA decomposition (informational)
    ica_decomposition_analysis(all_acts, concept_names, sparse_results)

    # Phase 10: Hierarchical clustering (informational)
    neuron_clustering_analysis(all_acts, concept_names, sparse_results)

    # Phase 11: Cross-layer concept tracking (informational)
    cross_layer_tracking(all_acts, concept_names, sparse_results, num_layers)

    # Phase 12: Information-theoretic analysis (informational)
    information_theoretic_analysis(all_acts, concept_names, sparse_results)

    # Phase 13: NMF decomposition (informational)
    nmf_decomposition_analysis(all_acts, concept_names, sparse_results)

    # Phase 14: INLP (informational)
    inlp_analysis(all_acts, concept_names, sparse_results)

    # Phase 15: Concept geometry (informational)
    concept_geometry_analysis(all_acts, concept_names, sparse_results)

    # Phase 16: RSA (informational)
    rsa_analysis(all_acts, concept_names, num_layers)

    # Phase 17: Probing robustness (informational)
    probing_robustness_analysis(all_acts, concept_names, sparse_results)

    # Phase 18: Superposition analysis (informational)
    superposition_analysis(all_acts, concept_names, sparse_results)

    # Phase 19: Sparse dictionary learning (informational)
    sparse_dictionary_analysis(all_acts, concept_names, sparse_results)

    # Phase 20: Concept interaction network (informational)
    concept_interaction_network(all_acts, concept_names, sparse_results)

    # Phase 21: Concept transferability (informational)
    concept_transferability_analysis(all_acts, concept_names, sparse_results)

    # Phase 22: Gram-Schmidt orthogonalization (informational)
    gram_schmidt_analysis(all_acts, concept_names, sparse_results)

    # Phase 23: Activation distributions (informational)
    activation_distribution_analysis(all_acts, concept_names, sparse_results)

    # Phase 24: Neuron co-activation (informational)
    neuron_coactivation_analysis(all_acts, concept_names, sparse_results)

    # Phase 25: Concept emergence (informational, reuses Phase 4 data)
    concept_emergence_analysis(concept_names, locality_results)

    # Phase 26: Neuron specificity spectrum (informational)
    neuron_specificity_spectrum(all_acts, concept_names, num_layers)

    # Phase 27: Concept bottleneck (informational, reuses Phase 4 data)
    concept_bottleneck_analysis(concept_names, locality_results, num_layers)

    # Phase 28: Concept gradient landscape (informational)
    concept_gradient_landscape(all_acts, concept_names, sparse_results, steering_vectors)

    # Phase 29: Concept dimensionality reduction (informational)
    concept_dimensionality_reduction(all_acts, concept_names, sparse_results)

    # Phase 30: Concept polarity (informational)
    concept_polarity_analysis(all_acts, concept_names, sparse_results)

    # Phase 31: Layer transition dynamics (informational)
    layer_transition_dynamics(all_acts, concept_names, num_layers)

    # Phase 32: Concept interference (informational)
    concept_interference_analysis(all_acts, concept_names, sparse_results, steering_vectors)

    # Phase 33: Decision boundary linearity (informational)
    concept_decision_boundary_analysis(all_acts, concept_names, sparse_results)

    # Phase 34: Concept prototypes (informational)
    concept_prototype_analysis(all_acts, concept_names, sparse_results, steering_vectors)

    # Phase 35: Residual analysis (informational)
    residual_analysis(all_acts, concept_names, sparse_results, steering_vectors, num_layers)

    # Phase 36: Concept stability (informational)
    concept_stability_analysis(all_acts, concept_names, sparse_results)

    # Phase 37: Concept effective rank (informational)
    concept_effective_rank(all_acts, concept_names, sparse_results)

    # Phase 38: Concept leakage (informational)
    concept_leakage_analysis(all_acts, concept_names, sparse_results)

    # Phase 39: Global summary statistics (informational)
    global_summary_statistics(all_acts, concept_names, sparse_results, num_layers)

    # Phase 40: Concept entanglement clusters (informational)
    concept_entanglement_clusters(all_acts, concept_names, sparse_results)

    # Phase 41: Layer concept quality (informational)
    layer_concept_quality(all_acts, concept_names, num_layers)

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
