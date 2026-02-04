# Self-Organizing Maps (SOMs)

*A comprehensive methodological and practical report for unsupervised analysis*

---

## 0. Scope and Philosophy

This document provides a **self-contained, technically rigorous introduction to Self‑Organizing Maps (SOMs)**, with emphasis on:

* Mathematical intuition
* Algorithmic structure
* Practical usage in high‑dimensional descriptive datasets
* Interpretation pitfalls
* Reproducible Python examples

The intended use case is **exploratory and descriptive analysis**, such as player game‑by‑game performance data, but the methodology is domain‑agnostic.

---

## 1. What Is a Self‑Organizing Map?

A **Self‑Organizing Map (SOM)** is an **unsupervised neural network** that maps high‑dimensional input data onto a **low‑dimensional (usually 2D) discrete grid**, while preserving the **topological structure** of the data.

Informally:

> Similar data points are mapped to nearby locations on the grid.

SOMs were introduced by **Teuvo Kohonen** in the early 1980s and occupy a conceptual space between:

* clustering algorithms (k‑means), and
* nonlinear dimensionality‑reduction techniques (PCA, t‑SNE, UMAP).

---

## 2. Core Concepts and Terminology

### 2.1 Neurons and Weight Vectors

Each node (neuron) on the SOM grid is associated with a **weight vector**:

[ \mathbf{w}_i \in \mathbb{R}^d ]

where *d* is the number of input features.

The grid itself is usually:

* rectangular or hexagonal
* two‑dimensional

---

### 2.2 Best Matching Unit (BMU)

Given an input vector ( \mathbf{x} ), the **Best Matching Unit** is the neuron whose weight vector is closest to ( \mathbf{x} ) under a chosen distance metric (typically Euclidean):

[ \text{BMU} = \arg\min_i | \mathbf{x} - \mathbf{w}_i | ]

---

### 2.3 Neighborhood Structure

Unlike k‑means, SOM updates **not only the BMU**, but also its neighbors on the grid.

This neighborhood structure is what enforces **topology preservation**.

---

## 3. Learning Algorithm

### 3.1 Training Loop

The SOM training algorithm proceeds iteratively:

1. Sample an input vector ( \mathbf{x} )
2. Identify its BMU
3. Update the BMU and its neighbors
4. Decrease learning rate and neighborhood radius

---

### 3.2 Weight Update Rule

For neuron ( i ) at iteration ( t ):

[
\mathbf{w}_i(t+1) = \mathbf{w}_i(t)

* \alpha(t), h_{bi}(t) , [\mathbf{x}(t) - \mathbf{w}_i(t)]
  ]

Where:

* ( \alpha(t) ) is the learning rate
* ( h_{bi}(t) ) is the neighborhood kernel centered on the BMU ( b )

Common neighborhood kernel:

[
h_{bi}(t) = \exp\left(- \frac{|r_b - r_i|^2}{2\sigma(t)^2} \right)
]

---

### 3.3 Convergence Behavior

Early training:

* Large ( \sigma )
* Coarse global organization

Late training:

* Small ( \sigma )
* Fine local adjustment

SOM training is **heuristic but stable** when inputs are standardized.

---

## 4. Why SOMs Are Different From Other Methods

| Method  | Structure Preserved     | Output             |
| ------- | ----------------------- | ------------------ |
| k‑means | None                    | Hard clusters      |
| PCA     | Variance                | Linear components  |
| t‑SNE   | Local similarity        | Visualization only |
| UMAP    | Local/global similarity | Visualization      |
| **SOM** | Topology                | Structured 2D map  |

Key distinction:

> SOMs preserve **neighborhood relationships**, not just distances.

---

## 5. Data Requirements and Preprocessing

### 5.1 Feature Scaling (Mandatory)

All SOM inputs must be **standardized**:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Unscaled data invalidates distance comparisons.

---

### 5.2 Redundancy Reduction

Highly correlated variables distort the map geometry.

Recommended practice:

* Correlation filtering (|ρ| > 0.85)
* Block‑wise PCA (optional)

---

### 5.3 Sample Interpretation

When samples are *(entity, time)* pairs (e.g., player‑game):

* Neurons represent **statistical states**, not identities
* Trajectories across the map are meaningful

---

## 6. SOMs in Practice: Python (MiniSom)

### 6.1 Installation

```bash
pip install minisom
```

---

### 6.2 Training a SOM

```python
import numpy as np
from minisom import MiniSom

som = MiniSom(
    x=12, y=12,
    input_len=X_scaled.shape[1],
    sigma=1.2,
    learning_rate=0.5,
    neighborhood_function='gaussian'
)

som.random_weights_init(X_scaled)
som.train_random(X_scaled, num_iteration=5000)
```

Grid size rule of thumb:

[
N_{neurons} \approx 5 \sqrt{N_{samples}}
]

---

## 7. Visualization and Interpretation

### 7.1 U‑Matrix (Unified Distance Matrix)

The U‑Matrix visualizes distances between neighboring neurons.

```python
import matplotlib.pyplot as plt

plt.imshow(som.distance_map().T, cmap='bone_r')
plt.colorbar(label='Inter‑Neuron Distance')
plt.title('SOM U‑Matrix')
plt.show()
```

Interpretation:

* Light regions → homogeneous clusters
* Dark ridges → cluster boundaries

---

### 7.2 Mapping Observations

```python
bmu_coordinates = np.array([som.winner(x) for x in X_scaled])
```

This allows:

* Density maps
* Trajectory analysis
* Group overlays

---

### 7.3 Component Planes

Component planes show how individual variables vary across the map.

```python
weights = som.get_weights()

for i, feature in enumerate(feature_names):
    plt.figure()
    plt.imshow(weights[:,:,i].T, cmap='coolwarm')
    plt.title(feature)
    plt.colorbar()
    plt.show()
```

---

## 8. Evaluation Metrics for SOMs

Standard clustering metrics are insufficient.

Useful SOM‑specific diagnostics:

* **Quantization Error**
* **Topographic Error**
* **Neighborhood Preservation**

These measure how well the SOM preserves input topology.

---

## 9. Strengths and Limitations

### Strengths

* Topology preservation
* Interpretability
* Continuous role structure
* Robust to noise

### Limitations

* No probabilistic model
* Hyperparameter sensitivity
* Slower than k‑means
* Visualization bias risk

---

## 10. Recommended References

### Foundational

* Kohonen, T. *Self‑Organizing Maps*. Springer.
* Kohonen, T. (1982). *Self‑organized formation of topologically correct feature maps*.

### Methodological

* Vesanto et al. (2000). *Clustering of the Self‑Organizing Map*.
* Forest et al. (2020). *Metrics for topology preservation in SOMs*.

### Tutorials & Implementations

* MiniSom (Python): JustGlowing GitHub repository
* Vesanto SOM Toolbox (MATLAB)

---

## 11. Final Remarks

SOMs should be treated as **exploratory cartographic tools**, not classifiers.

A successful SOM analysis answers:

> “What are the dominant structures in the data, and how do observations move between them?”

—not:

> “Which cluster is best?”

---

*End of document.*
