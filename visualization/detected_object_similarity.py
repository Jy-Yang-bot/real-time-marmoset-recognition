"""
Evaluation of the similarity between detected label classes, based on the embedding of the trained YOLO model.
- Input: images for object detection, trained weight from models.
- Output: heat map, violin plots of the similarity.
- In this program, we compute the similarity of marmoset faces between different family relationship pairs.
- For example, here we evaluate the relationships from 2 models, 1 family and 1 twin.

Contributor: Jiayue Yang, 2025-12-02
"""
# import libraries
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

# define the inputs
# example: the detection labels = detected marmoset identity
image_files = "dir_images_of_detection" # (to edit) the directory of the images to be detected using the trained model
# weights of the trained model, and the relationships between the detected label classes
models = {
    "relationship type": { # (to edit) relationship types of the model (e.g. family, couples, etc.)
        "model_path": "weight.pt", # (to edit) the directory of the trained weight from the model
        "subjects": {
            "A": "family_role", # (to edit) detection labels and their corresponding family role
            "B": "family_role", # (to edit) detection labels and their corresponding family role
            "C": "family_role", # (to edit) detection labels and their corresponding family role
        },
        # the pairs of detection labels to assess
        "pairs": [
            ("family_role", "family_role"), # (to edit) relationship pairs between multiple detection labels
            ("family_role", "family_role"),
            ("family_role", "family_role"),
        ]
    },
    # if wanting to assess multiple models, can add here
    # the format of the model should be the same as the example above
}

# define functions to extract model embeddings, and statistical tests
# funtion 1: extract embeddings from the model, with application on the images with detected objects
def extract_embeddings(model, image_paths):
    embs = []
    with torch.no_grad():
        for p in image_paths:
            # extract the trained embeddings from specific model and apply them on the images with detected objects
            e = model.embed(p)[0].cpu().numpy().squeeze()
            embs.append(e)
    return np.vstack(embs)

# function 2: compute the effect size of 2 arrays
def compute_cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (np.mean(x) - np.mean(y)) / s_pooled

# function 3: run the t test and the effect size (using function 2 here)
def run_within_model_stats(df, value_col):
    results = []
    # for each model (as model may differ in embedding sizes, the stat tests are performed within each model)
    for model_name in df["Model"].unique():
        df_m = df[df["Model"] == model_name].copy()
        # compute z score for normalization within model
        df_m[value_col + "_z"] = (df_m[value_col] - df_m[value_col].mean()) / df_m[value_col].std(ddof=1)
        rels = df_m["Relationship"].unique()
        # iterate through different relationship pairs
        for i in range(len(rels)):
            # compute the Welch's t-test and cohen's d to assess the significance in differenec and effect sizes between 2 pairs of relationship
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

# main run --> compute the similarity between detected labels
# other similarity assessment can be used, in the program, we used cosine similarity and Euclidean distance
# create empty lists for storing cos similarity and euc distance
all_cos_rows = []
all_euc_rows = []

# iterate through models of input
for model_name, cfg in models.items():
    print(f"\n=== Processing {model_name} model ===")
    model = YOLO(cfg["model_path"])
    model.eval()
    # empty array to store embedings
    all_embeddings = {}
    mean_embeddings = {}

    # load images from the file and extract embeddings
    for prefix, label in cfg["subjects"].items():
        image_files = [
            os.path.join(image_files, f)
            for f in os.listdir(image_files)
            if f.lower().startswith(prefix.lower())
            and f.lower().endswith((".jpg", ".png"))
        ]
        # display the number of images per label/subject
        print(f"{label}: {len(image_files)} images")
        # extract and then normalize the embeddings
        embeds = extract_embeddings(model, image_files)
        embeds = normalize(embeds, axis=1)
        # compute the mean embeddings for each label/subject
        all_embeddings[label] = embeds
        mean_embeddings[label] = embeds.mean(axis=0)

    # generate heat maos to visualize the similarity
    subjects = list(mean_embeddings.keys())

    # compute cosine similarity
    cos_mat = np.zeros((len(subjects), len(subjects)))
    for i, s1 in enumerate(subjects):
        for j, s2 in enumerate(subjects):
            cos_mat[i, j] = 1 - cosine(mean_embeddings[s1], mean_embeddings[s2])
    # plot the calculated results
    plt.figure(figsize=(5, 4))
    sns.heatmap(cos_mat, xticklabels=subjects, yticklabels=subjects, annot=True, cmap="viridis")
    plt.title(f"{model_name} – Cosine Similarity")
    plt.tight_layout()
    plt.show()

    # compute Euclidean distance
    euc_mat = np.zeros_like(cos_mat)
    for i, s1 in enumerate(subjects):
        for j, s2 in enumerate(subjects):
            euc_mat[i, j] = np.linalg.norm(mean_embeddings[s1] - mean_embeddings[s2])
    # plot the calculated results
    plt.figure(figsize=(5, 4))
    sns.heatmap(euc_mat, xticklabels=subjects, yticklabels=subjects, annot=True, cmap="magma_r")
    plt.title(f"{model_name} – Euclidean Distance")
    plt.tight_layout()
    plt.show()

    # assess the cosine similarity and euclidean distances between 2 individuals in the pre-defined pairs
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

# plot the similarity across diffferent relationship pairs
df_cos = pd.DataFrame(all_cos_rows)
df_euc = pd.DataFrame(all_euc_rows)

# (just for visualization, optional) can normalize the similarity score within the model
df_cos["Cosine_z"] = df_cos.groupby("Model")["Cosine Similarity"].transform(lambda x: (x - x.mean())/x.std(ddof=1))
df_euc["Euclidean_z"] = df_euc.groupby("Model")["Euclidean Distance"].transform(lambda x: (x - x.mean())/x.std(ddof=1))

# plot the figure of cos similarity
plt.figure(figsize=(7, 4))
sns.violinplot(data=df_cos, x="Relationship", y="Cosine_z", inner="box", cut=0)
plt.xlabel("Relationship", fontsize=18)
plt.ylabel("Cosine Similarity (normalized z-score)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Face Similarity Between Family Relationships (Cosine Similarity)", fontsize=22)
plt.tight_layout()
plt.show()
# plot the figure of euclidean distance
plt.figure(figsize=(7, 4))
sns.violinplot(data=df_euc, x="Relationship", y="Euclidean_z", inner="box", cut=0)
plt.xlabel("Relationship", fontsize=18)
plt.ylabel("Euclidean Distance (normalized z-score)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Face Similarity Between Family Relationships (Euclidean Distance)", fontsize=22)
plt.tight_layout()
plt.show()

# statistical tests
print("\n===== Cosine Similarity Statistical Tests (within-model) =====")
cos_stats = run_within_model_stats(df_cos, "Cosine Similarity")
print(cos_stats)

print("\n===== Euclidean Distance Statistical Tests (within-model) =====")
euc_stats = run_within_model_stats(df_euc, "Euclidean Distance")
print(euc_stats)

# (optional) save embedding results
np.save("mean_identity_embeddings.npy", all_embeddings)
np.save("all_embeddings.npy", all_embeddings)

print("Similarity assessment completed: please see the figures generated and the statistical tests to examine for significance.")


