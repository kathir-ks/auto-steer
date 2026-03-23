# Mathematical Concepts Used in Auto-Steer

A comprehensive inventory of every mathematical concept, statistical method, metric, and technique employed across the ~2,390 analysis phases in `steer.py`.

**Total: ~150+ distinct mathematical concepts** spanning linear algebra, statistics, information theory, machine learning, signal processing, geometry, and representational analysis.

---

## 1. Linear Algebra & Matrix Operations

| Concept | What It Does Here |
|---------|------------------|
| Singular Value Decomposition (SVD) | Decompose activation matrices to find principal directions |
| Principal Component Analysis (PCA) | Reduce dimensionality, find variance structure |
| Eigenvalue/Eigenvector decomposition | Analyze covariance and Gram matrices |
| Gram-Schmidt orthogonalization | Remove concept overlap, decorrelate directions |
| Gram matrix & determinant | Measure linear independence of concept directions |
| Null space projection (INLP) | Iteratively erase concept info to test robustness |
| Matrix rank | Check dimensionality of concept subspaces |
| Condition number | Measure numerical stability of concept representation |
| Effective rank | Spectral-entropy-based "soft" rank of matrices |
| Participation ratio | How many eigenvalues contribute meaningfully |
| Covariance matrix | Capture co-variation between neurons |
| Correlation matrix | Normalize covariance to unit scale |
| Orthogonal projection | Project activations onto concept directions |
| Orthogonal complement | Residual after removing concept signal |
| Whitening / ZCA whitening | Decorrelate features before angle measurements |
| Matrix trace | Sum of eigenvalues, total variance |

---

## 2. Distance & Similarity Metrics

| Metric | Use |
|--------|-----|
| Cosine similarity/distance | Primary measure of direction alignment between concept vectors |
| Euclidean distance (L2) | Pairwise distances between activation vectors |
| L1 norm (Manhattan) | Sparsity measurement |
| Mahalanobis distance | Class separation accounting for covariance structure |
| Bhattacharyya distance | Overlap between two distributions |
| Wasserstein distance (Earth Mover's) | Distribution comparison across layers |
| Hellinger distance | Symmetric distribution divergence |
| Jensen-Shannon divergence | Symmetric variant of KL divergence |
| KL divergence (Kullback-Leibler) | Asymmetric information divergence between distributions |
| Jaccard index/distance | Set overlap between top-K neuron sets |
| Geodesic distance | Distance along the manifold surface |
| Spearman rank correlation | Monotonic relationship between rankings |
| Pearson correlation | Linear relationship between variables |
| Point-biserial correlation | Correlation between binary label and continuous activation |
| Angular distance | Arc-cosine of cosine similarity |
| Principal angles | Angles between two subspaces |

---

## 3. Information Theory

| Concept | Use |
|---------|-----|
| Shannon entropy | Uncertainty in neuron activations or importance distributions |
| Differential entropy | Continuous-variable entropy |
| Mutual information (MI) | Shared information between neuron activations and concept labels |
| Conditional entropy | Remaining uncertainty after observing another variable |
| Joint entropy | Combined uncertainty of two variables |
| Information gain | MI framed as entropy reduction |
| Information bottleneck | Layer-wise compression/retention of concept info |
| Channel capacity | Maximum transmittable information per neuron |
| Bits per neuron | Information content normalized by neuron count |
| Cross-entropy | Loss function perspective on classification |
| Information density | Information per unit of representation |

---

## 4. Statistical Distributions & Moments

| Concept | Use |
|---------|-----|
| Mean, median, mode | Central tendency of activations |
| Variance / standard deviation | Spread of activations |
| Skewness (3rd moment) | Asymmetry of activation distributions |
| Kurtosis (4th moment) | Tail heaviness of distributions |
| Higher moments | Beyond 4th moment characterization |
| Bimodality coefficient | Whether activations cluster into two groups |
| Interquartile range (IQR) | Robust spread measure |
| Mean absolute deviation (MAD) | Robust alternative to standard deviation |
| Quantiles / percentiles | Distribution characterization |
| Histogram analysis | Binned distribution shape |
| Normality testing (Shapiro-Wilk) | Whether activations are Gaussian |
| Power law fitting | Whether neuron importance follows power-law decay |

---

## 5. Effect Size & Statistical Testing

| Method | Use |
|--------|-----|
| Cohen's d | Effect size measuring how much a neuron differentiates concept vs non-concept |
| d-prime (d') | Signal detection theory discriminability |
| Fisher discriminant ratio | Between-class vs within-class variance |
| F-ratio (ANOVA) | Variance explained by concept grouping |
| Point-biserial correlation | Binary-continuous association strength |
| Bootstrap analysis | Resampling-based confidence estimation |
| Jackknife (leave-one-out) | Bias estimation by systematic sample removal |
| Permutation testing | Null distribution by label shuffling |
| Split-half reliability | Consistency across random data splits |
| Confidence intervals | Uncertainty bounds on estimates |
| Kolmogorov-Smirnov test | Distribution comparison |
| t-test (implicit via Cohen's d) | Mean difference significance |

---

## 6. Classification & Machine Learning

| Method | Use |
|--------|-----|
| Logistic regression (L2) | Linear probes for concept classification |
| Logistic regression (L1 / Lasso) | Sparse probes — forces neuron selection |
| Cross-validation (stratified k-fold) | Unbiased accuracy estimation |
| Leave-one-out cross-validation | Exhaustive CV for small budgets |
| ROC / AUC | Classification performance across thresholds |
| Balanced accuracy | Accuracy correcting for class imbalance |
| Decision boundaries | Where classifier transitions between classes |
| SVM margin | Maximum-margin separation |
| Neuron budget sweeps | Accuracy vs number of neurons curves |

---

## 7. Decomposition & Feature Extraction

| Method | Use |
|--------|-----|
| Independent Component Analysis (ICA) | Find statistically independent components in activations |
| Non-negative Matrix Factorization (NMF) | Parts-based decomposition (non-negative constraints) |
| Sparse Dictionary Learning | Learn overcomplete sparse basis (SAE-inspired) |
| Sparse coding coefficients | How activations decompose over learned dictionary |
| Reconstruction error | Quality of decomposition |
| Basis pursuit | Finding sparsest representation in overcomplete basis |

---

## 8. Representational Analysis Methods

| Method | Use |
|--------|-----|
| Representational Similarity Analysis (RSA) | Compare representation structure across layers |
| Representational Dissimilarity Matrix (RDM) | Pairwise dissimilarity structure |
| Centered Kernel Alignment (CKA) | Layer similarity comparison |
| INLP (Iterative Nullspace Projection) | Erase concept info iteratively to test encoding depth |
| Activation patching | Swap activations between samples to test causal role |
| Steering vector injection | Add concept direction to test controllability |

---

## 9. Concentration & Inequality Metrics

| Metric | Use |
|--------|-----|
| Gini coefficient | Inequality/concentration of layer accuracy or neuron importance |
| Herfindahl-Hirschman Index (HHI) | Concentration of importance across neurons |
| Hoyer sparsity | Ratio of L1/L2 norms as sparsity measure |
| L1/L2 sparsity ratio | Alternative sparsity characterization |
| Top-K concentration | Fraction of signal in top-K neurons |
| Power-mean aggregation | Generalized mean (p=2, p=3) for locality scoring |
| Participation ratio | Effective number of contributing components |

---

## 10. Geometric & Topological Concepts

| Concept | Use |
|---------|-----|
| Subspace angles | Angles between concept subspaces |
| Angular velocity | Rate of direction change across layers |
| Angular dispersion | Spread of angles in a set of directions |
| Curvature | Rate of direction change along layer depth |
| Convex hull | Bounding geometry of activation clouds |
| Isotropy | Whether activations are uniformly distributed in all directions |
| Anisotropy | Directional bias in activation space |
| Centroid analysis | Mean position of concept activation clouds |
| Manifold dimensionality | Intrinsic dimension of activation manifold |

---

## 11. Neuron-Level Metrics

| Metric | Use |
|--------|-----|
| Neuron importance ranking | Order neurons by discriminative power |
| Selectivity index | How specific a neuron is to one concept |
| Monosemanticity score | 1-to-1 mapping quality between neurons and concepts |
| Disjointness score | Whether neuron sets for different concepts overlap |
| Firing rate | Fraction of samples activating a neuron above threshold |
| Dead neuron detection | Neurons with zero or near-zero variance |
| Neuron saturation | Neurons stuck at extreme values |
| Co-activation analysis | Which neurons fire together |
| Neuron functional types | Specialist / hub / generalist / silent classification |
| Redundancy index | How replaceable a neuron is |

---

## 12. Robustness & Stability Methods

| Method | Use |
|--------|-----|
| Dropout robustness | Accuracy after random neuron removal |
| Noise injection | Stability under Gaussian perturbation |
| Subsampling stability | Consistency with fewer training samples |
| Direction stability | Cosine similarity of concept directions across subsets |
| Rank stability | Whether neuron rankings are consistent across splits |
| Sample size sensitivity | How many samples needed for stable estimates |

---

## 13. Signal Processing & Energy Concepts

| Concept | Use |
|---------|-----|
| Signal-to-noise ratio (SNR) | Concept signal vs background noise |
| Signal energy | Total squared activation magnitude |
| Energy distribution | How signal energy spreads across layers |
| Dynamic range | Ratio of max to min activation magnitudes |
| Spectral gap | Gap between leading eigenvalues |
| Eigenspectrum analysis | Full eigenvalue distribution shape |
| Power spectral density | Frequency-domain characterization |
| Signal decay | How concept signal attenuates across layers |
| Bandwidth | Range of layers with strong signal |
| FWHM (Full Width at Half Maximum) | Peak width of layer-wise accuracy curve |

---

## 14. Variance Decomposition

| Method | Use |
|--------|-----|
| Between-class / within-class variance | Fisher-style separation |
| Variance explained ratio | Fraction of variance captured by concept directions |
| Residual variance | Variance not explained by concepts |
| Cumulative variance explained | PCA scree analysis |
| Variance partition | Splitting total variance into components |

---

## 15. Normalization & Preprocessing

| Method | Use |
|--------|-----|
| Z-scoring / StandardScaler | Normalize features to zero mean, unit variance |
| Whitening | Remove correlations between features |
| Mean centering | Subtract mean before analysis |
| Norm scaling | Divide by vector norm |
| Feature normalization | Per-feature scaling |

---

## Libraries Used

| Library | Functions |
|---------|-----------|
| **NumPy** | `linalg.svd`, `linalg.eigvalsh`, `linalg.norm`, `linalg.det`, `linalg.matrix_rank`, `linalg.inv`, `outer`, `dot`, `eye` |
| **SciPy (spatial)** | `pdist`, `squareform`, cosine distances |
| **SciPy (cluster)** | `linkage` (hierarchical clustering), `fcluster` |
| **SciPy (stats)** | `skew`, `kurtosis`, `spearmanr` |
| **scikit-learn (decomposition)** | `PCA`, `FastICA`, `NMF`, `DictionaryLearning` |
| **scikit-learn (linear_model)** | `LogisticRegression` (L1 and L2 penalties) |
| **scikit-learn (model_selection)** | `cross_val_score`, `StratifiedKFold` |
| **scikit-learn (preprocessing)** | `StandardScaler` |
| **scikit-learn (feature_selection)** | `mutual_info_classif` |
