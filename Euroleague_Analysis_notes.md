# Unsupervised Learning for EuroLeague Player Game-by-Game Analysis

## Scope of This Report

This document provides methodological notes and practical guidance for conducting an **unsupervised analysis** of EuroLeague basketball player **game-by-game** statistics. The focus is *descriptive and exploratory*, not predictive. The goal is to discover **latent structures**, **player archetypes**, and **statistical regularities** without labels.

---

## 1. Notes on Unsupervised Learning (Contextualized for Sports Analytics)

### 1.1 What Unsupervised Learning Is (and Is Not)

Unsupervised learning aims to identify **patterns, clusters, manifolds, or low-dimensional representations** in data **without predefined targets**.

In this setting, it is used to:

* Discover *player roles* rather than evaluate performance
* Identify *statistical archetypes*, not talent rankings
* Explore *structure*, not causality

It is **not**:

* A replacement for tactical analysis
* A measure of player value
* A causal inference tool

A safe guiding principle:

> Unsupervised learning tells you *how players are statistically similar*, not *why*.

---

### 1.2 Typical Unsupervised Tasks Relevant Here

* **Clustering**: grouping players by statistical similarity
* **Dimensionality Reduction**: revealing dominant axes of variation
* **Topology Preservation**: mapping continuous transitions between roles
* **Outlier Detection**: identifying statistically rare player-games

Common algorithms:

* k-Means / k-Medoids
* Hierarchical clustering
* PCA / ICA
* t-SNE / UMAP
* Self-Organizing Maps (SoM)

---

### 1.3 Unit of Analysis Matters

You are working at the level of:

> **(player, game)** observations

This implies:

* The same player appears multiple times
* Variability within a player is as important as variability across players
* Clusters may represent *roles in games*, not fixed player identities

This is a feature, not a bug.

---

## 2. Directions to Enhance Your Dataset (Before Any Algorithm)

### 2.1 Normalize Aggressively and Transparently

Mandatory transformations:

* Per-minute or per-possession scaling
* Pace-adjusted metrics
* Usage-normalized efficiency

Document explicitly:

> All variables are standardized per game and per opportunity.

---

### 2.2 Separate Variable Families

Create **semantic blocks** of variables:

* **Usage & Volume** (USG%, FGA, touches)
* **Efficiency** (TS%, PPS, ORtg)
* **Playmaking** (AST%, potential assists)
* **Defense** (stocks, DRtg proxies)
* **Rebounding context** (ORB%, DRB%)
* **Stability** (rolling SDs, volatility indices)

This allows:

* Block-wise PCA
* Weighted distance metrics
* Interpretability of clusters

---

### 2.3 Reduce Redundancy Before Clustering

Highly correlated variables distort distance-based methods.

Recommended steps:

1. Compute correlation matrix
2. Remove or aggregate variables with |ρ| > 0.85
3. Prefer *interpretable representatives*

Example:

* Keep TS%, drop FG% and eFG%

---

### 2.4 Encode Context Explicitly

Add **context variables**, even if crude:

* Home vs away (binary)
* Opponent strength proxy
* Minutes played that game
* Game pace

These often explain cluster separation more than skill variables.

---

### 2.5 Standardization Is Non-Negotiable

Before *any* unsupervised method:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Distance-based methods without scaling are statistically meaningless.

---

## 3. Useful Python Plotting Functions (Exploratory Phase)

### 3.1 Correlation Matrix Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.show()
```

Use this **before** dimensionality reduction.

---

### 3.2 Clustered Heatmap (Variables + Observations)

```python
sns.clustermap(df, method='ward', metric='euclidean',
               standard_scale=1, cmap='viridis')
plt.show()
```

Excellent for visually inspecting emergent structure.

---

### 3.3 PCA Explained Variance Plot

```python
from sklearn.decomposition import PCA
import numpy as np

pca = PCA()
pca.fit(X_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
```

This tells you **how many dimensions matter**.

---

### 3.4 2D Embeddings (t-SNE / UMAP)

```python
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2, perplexity=30).fit_transform(X_scaled)

plt.scatter(X_embedded[:,0], X_embedded[:,1], s=10)
plt.title("t-SNE embedding")
plt.show()
```

Use *only for visualization*, not clustering.

---

## 4. Self-Organizing Maps (SoM)

### 4.1 What Is a Self-Organizing Map?

A **Self-Organizing Map** is a neural-network–based unsupervised method that:

* Projects high-dimensional data onto a 2D grid
* Preserves topological relationships
* Creates a *continuous map of similarity*

Unlike k-means, SoM:

* Does not force hard clusters
* Reveals smooth transitions between archetypes

Originally introduced by **Teuvo Kohonen (1982)**.

---

### 4.2 Why SoM Is Well-Suited Here

In player game-by-game analysis:

* Roles are *continuous*, not discrete
* Players drift between roles across games
* Border cases are informative

SoM excels at exactly this structure.

---

### 4.3 Typical Interpretation

* Each neuron = statistical archetype
* Nearby neurons = similar player-games
* Dense regions = common roles
* Sparse regions = rare profiles

You can:

* Map individual players over time
* Study role variability
* Compare teams stylistically

---

### 4.4 Minimal Python Example (MiniSom)

```python
from minisom import MiniSom

som = MiniSom(x=10, y=10, input_len=X_scaled.shape[1],
              sigma=1.0, learning_rate=0.5)

som.random_weights_init(X_scaled)
som.train_random(X_scaled, num_iteration=5000)
```

Visualization:

```python
plt.imshow(som.distance_map().T, cmap='bone_r')
plt.colorbar(label='U-Matrix')
plt.title('SoM U-Matrix')
plt.show()
```

The **U-Matrix** highlights cluster boundaries naturally.

---

## Final Remarks

* Treat unsupervised analysis as *cartography*, not judgment
* Document every preprocessing choice
* Prefer interpretability over algorithmic sophistication
* Expect ambiguity—and embrace it

A good unsupervised result should make you say:

> "That makes sense—and I hadn’t thought of it that way."

---

If you want next steps:

* A **full SoM-based workflow** tailored to EuroLeague data
* A **comparison: PCA vs UMAP vs SoM** for this exact problem
* A **player trajectory analysis** across the season

Just say the word.
