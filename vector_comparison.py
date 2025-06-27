import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Load embeddings and labels
X_en = np.load("results_english/doc_embeddings.npy")
X_fr = np.load("results_french/doc_embeddings.npy")
labels_en = pd.read_csv("results_english/labels.csv").squeeze()
labels_fr = pd.read_csv("results_french/labels.csv").squeeze()

print(f"X_en: {len(X_en)}, X_fr: {len(X_fr)}")
print(f"labels_en: {len(labels_en)}, labels_fr: {len(labels_fr)}")

# Ensure correspondence (assumes aligned ordering)
assert len(X_en) == len(X_fr) == len(labels_en) == len(labels_fr)
assert all(labels_en == labels_fr), "Mismatch in genre labels between EN and FR"

labels = labels_en  # either is fine now

# Compute centroids: EN-based for most, FR-based for Poems
genres = labels.unique()
centroids = {}

for genre in genres:
    mask = labels == genre
    if genre == "Poems":
        centroid = X_fr[mask].mean(axis=0)
    else:
        centroid = X_en[mask].mean(axis=0)
    centroids[genre] = centroid.reshape(1, -1)

# Compare each doc (EN & FR) to its genre centroid
results = []

for i in range(len(labels)):
    genre = labels.iloc[i]
    centroid = centroids[genre]

    sim_en = cosine_similarity([X_en[i]], centroid)[0][0]
    sim_fr = cosine_similarity([X_fr[i]], centroid)[0][0]

    results.append({
        "index": i,
        "genre": genre,
        "similarity_EN_to_centroid": sim_en,
        "similarity_FR_to_centroid": sim_fr,
        "difference_FR_minus_EN": sim_fr - sim_en
    })

df_results = pd.DataFrame(results)

# Save results
os.makedirs("results_vectors", exist_ok=True)
df_results.to_csv("results_vectors/translation_semantic_drift.csv", index=False)
print("✅ Saved: results_vectors/translation_semantic_drift.csv")

#-------------------------

# Load the CSV we saved earlier
df = pd.read_csv("results_vectors/translation_semantic_drift.csv")

# Plot average drift per genre
plt.figure(figsize=(10, 6))
order = df.groupby("genre")["difference_FR_minus_EN"].mean().sort_values().index
sns.barplot(data=df, x="genre", y="difference_FR_minus_EN", order=order, estimator="mean", errorbar="sd", palette="viridis")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Average Semantic Drift by Genre (FR - EN Similarity to Centroid)")
plt.ylabel("Drift (↑ = closer to centroid in FR)")
plt.xlabel("Genre")
plt.tight_layout()
plt.savefig("results_vectors/semantic_drift_barplot.png")

#------------------------

# Reduce to 2D with PCA
pca = PCA(n_components=2)
all_vecs = np.vstack([X_en, X_fr])
reduced = pca.fit_transform(all_vecs)

n = len(X_en)
X_en_2d = reduced[:n]
X_fr_2d = reduced[n:]

# Plot arrows from EN to FR
plt.figure(figsize=(10, 8))
for i in range(n):
    plt.arrow(X_en_2d[i, 0], X_en_2d[i, 1],
              X_fr_2d[i, 0] - X_en_2d[i, 0],
              X_fr_2d[i, 1] - X_en_2d[i, 1],
              color='gray', alpha=0.4, head_width=0.05, length_includes_head=True)

# Color points by genre
for genre in df["genre"].unique():
    idxs = df[df["genre"] == genre].index
    plt.scatter(X_en_2d[idxs, 0], X_en_2d[idxs, 1], label=f"{genre}", marker='o', alpha=0.7)
    plt.scatter(X_fr_2d[idxs, 0], X_fr_2d[idxs, 1], label=f"{genre} (FR)", marker='^', alpha=0.7)

plt.title("PCA Projection of Document Embeddings (EN → FR)")
plt.xlabel("PC1")
plt.ylabel("PC2")

handles, labels_ = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels_, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best')

plt.tight_layout()
plt.savefig("results_vectors/semantic_drift_pca.png")
plt.close()
