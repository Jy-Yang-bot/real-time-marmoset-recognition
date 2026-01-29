# =====================================================
# Face similarity pipeline (Family + Twin models)
# =====================================================

import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine, cdist
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# =========================
# CONFIG
# =========================
IMAGE_ROOT = "C:\\Users\\jiayue\\Desktop\\faces"

MODELS = {
    "Family": {
        "model_path": "C:\\Users\\jiayue\\Desktop\\faces\\best_family.pt",
        "subjects": {
            "Adult1": "Father",
            "Adult2": "Mother",
            "Adult3": "Son",
        },
        "pairs": [
            ("Mother", "Father"),
            ("Father", "Son"),
            ("Mother", "Son"),
        ]
    },
    "Twin": {
        "model_path": "C:\\Users\\jiayue\\Desktop\\faces\\best_twin.pt",
        "subjects": {
            "Young1": "Twin1",
            "Young2": "Twin2",
        },
        "pairs": [
            ("Twin1", "Twin2"),
        ]
    }
}

# =========================
# FUNCTIONS
# =========================
def extract_embeddings(model, image_paths):
    embs = []
    with torch.no_grad():
        for p in image_paths:
            e = model.embed(p)[0].cpu().numpy().squeeze()
            embs.append(e)
    return np.vstack(embs)

def compute_cohens_d(x, y):
    """Compute Cohen's d for two arrays."""
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (np.mean(x) - np.mean(y)) / s_pooled

def run_within_model_stats(df, value_col):
    """Run model-aware t-tests and compute effect size (Cohen's d)."""
    results = []
    for model_name in df["Model"].unique():
        df_m = df[df["Model"] == model_name].copy()
        # z-score normalization within model
        df_m[value_col + "_z"] = (df_m[value_col] - df_m[value_col].mean()) / df_m[value_col].std(ddof=1)

        rels = df_m["Relationship"].unique()
        for i in range(len(rels)):
            for j in range(i + 1, len(rels)):
                r1, r2 = rels[i], rels[j]
                v1 = df_m[df_m["Relationship"] == r1][value_col + "_z"]
                v2 = df_m[df_m["Relationship"] == r2][value_col + "_z"]
                t, p = ttest_ind(v1, v2, equal_var=False)
                d = compute_cohens_d(v1, v2)
                results.append({
                    "Model": model_name,
                    "Comparison": f"{r1} vs {r2}",
                    "t statistic": t,
                    "p-value": p,
                    "Cohen's d": d
                })
    return pd.DataFrame(results)

# =========================
# MAIN LOOP (MULTI-MODEL)
# =========================
all_cos_rows = []
all_euc_rows = []

for model_name, cfg in MODELS.items():
    print(f"\n=== Processing {model_name} model ===")

    model = YOLO(cfg["model_path"])
    model.eval()

    all_embeddings = {}
    mean_embeddings = {}

    # --- Load images & extract embeddings ---
    for prefix, label in cfg["subjects"].items():
        image_files = [
            os.path.join(IMAGE_ROOT, f)
            for f in os.listdir(IMAGE_ROOT)
            if f.lower().startswith(prefix.lower())
            and f.lower().endswith((".jpg", ".png"))
        ]
        print(f"{label}: {len(image_files)} images")

        embeds = extract_embeddings(model, image_files)
        embeds = normalize(embeds, axis=1)

        all_embeddings[label] = embeds
        mean_embeddings[label] = embeds.mean(axis=0)

    # =========================
    # HEATMAPS (MEAN EMBEDDINGS)
    # =========================
    subjects = list(mean_embeddings.keys())

    # Cosine
    cos_mat = np.zeros((len(subjects), len(subjects)))
    for i, s1 in enumerate(subjects):
        for j, s2 in enumerate(subjects):
            cos_mat[i, j] = 1 - cosine(mean_embeddings[s1], mean_embeddings[s2])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cos_mat, xticklabels=subjects, yticklabels=subjects, annot=True, cmap="viridis")
    plt.title(f"{model_name} – Cosine Similarity")
    plt.tight_layout()
    plt.show()

    # Euclidean
    euc_mat = np.zeros_like(cos_mat)
    for i, s1 in enumerate(subjects):
        for j, s2 in enumerate(subjects):
            euc_mat[i, j] = np.linalg.norm(mean_embeddings[s1] - mean_embeddings[s2])

    plt.figure(figsize=(5, 4))
    sns.heatmap(euc_mat, xticklabels=subjects, yticklabels=subjects, annot=True, cmap="magma_r")
    plt.title(f"{model_name} – Euclidean Distance")
    plt.tight_layout()
    plt.show()

    # =========================
    # PER-IMAGE DISTRIBUTIONS
    # =========================
    for s1, s2 in cfg["pairs"]:
        cos_vals = 1 - cdist(
            all_embeddings[s1],
            mean_embeddings[s2][None, :],
            metric="cosine"
        ).flatten()

        euc_vals = cdist(
            all_embeddings[s1],
            mean_embeddings[s2][None, :],
            metric="euclidean"
        ).flatten()

        for v in cos_vals:
            all_cos_rows.append({
                "Model": model_name,
                "Relationship": f"{s1}–{s2}",
                "Cosine Similarity": v
            })

        for v in euc_vals:
            all_euc_rows.append({
                "Model": model_name,
                "Relationship": f"{s1}–{s2}",
                "Euclidean Distance": v
            })

# =========================
# COMBINED VIOLIN PLOTS (Z-SCORE)
# =========================
df_cos = pd.DataFrame(all_cos_rows)
df_euc = pd.DataFrame(all_euc_rows)

# Z-score normalization per model for plotting
df_cos["Cosine_z"] = df_cos.groupby("Model")["Cosine Similarity"].transform(lambda x: (x - x.mean())/x.std(ddof=1))
df_euc["Euclidean_z"] = df_euc.groupby("Model")["Euclidean Distance"].transform(lambda x: (x - x.mean())/x.std(ddof=1))


plt.figure(figsize=(7, 4))
sns.violinplot(data=df_cos, x="Relationship", y="Cosine_z", inner="box", cut=0)
plt.xlabel("Relationship", fontsize=18)
plt.ylabel("Cosine Similarity (normalized z-score)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Face Similarity Between Family Relationships (Cosine Similarity)", fontsize=22)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.violinplot(data=df_euc, x="Relationship", y="Euclidean_z", inner="box", cut=0)
plt.xlabel("Relationship", fontsize=18)
plt.ylabel("Euclidean Distance (normalized z-score)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Face Similarity Between Family Relationships (Euclidean Distance)", fontsize=22)
plt.tight_layout()
plt.show()

# =========================
# STATISTICS (WITHIN-MODEL)
# =========================
print("\n===== Cosine Similarity Statistical Tests (within-model) =====")
cos_stats = run_within_model_stats(df_cos, "Cosine Similarity")
print(cos_stats)

print("\n===== Euclidean Distance Statistical Tests (within-model) =====")
euc_stats = run_within_model_stats(df_euc, "Euclidean Distance")
print(euc_stats)

# =========================
# SAVE RESULTS
# =========================
np.save("mean_identity_embeddings.npy", all_embeddings)
np.save("all_embeddings.npy", all_embeddings)

print("\n✔ Finished: Family + Twin analyses complete with z-scores and effect sizes.")


# =========================
# Z-SCORE + RAW STATS SUMMARY (FOR INTERPRETATION)
# =========================

def print_zscore_summary(df, raw_col, z_col):
    print(f"\n===== {raw_col} Z-score summary (within-model) =====")
    summary = (
        df
        .groupby(["Model", "Relationship"])
        .agg(
            Raw_Mean=(raw_col, "mean"),
            Raw_SD=(raw_col, "std"),
            Z_Mean=(z_col, "mean"),
            Z_SD=(z_col, "std"),
            N=(raw_col, "count")
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    return summary


cos_summary = print_zscore_summary(
    df_cos,
    raw_col="Cosine Similarity",
    z_col="Cosine_z"
)

euc_summary = print_zscore_summary(
    df_euc,
    raw_col="Euclidean Distance",
    z_col="Euclidean_z"
)
