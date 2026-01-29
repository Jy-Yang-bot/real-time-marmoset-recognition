import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# User parameters
# =========================
csv_path = "results_young.csv"     # path to YOLO results CSV
output_dir = "figures2"       # folder to save figures
dpi = 300                    # figure resolution
best_epoch = 245             # epoch of best model selection
line_color = "darkred"       # color for vertical line
line_style = "--"            # dashed line

os.makedirs(output_dir, exist_ok=True)

# =========================
# Load CSV
# =========================
df = pd.read_csv(csv_path)
epochs = df["epoch"]

# =========================
# Panel A: Precision vs Epoch
# =========================
plt.figure(figsize=(5, 4))
plt.plot(epochs, df["metrics/precision(B)"], label="Precision", linewidth=1.5)
plt.axvline(x=best_epoch, color=line_color, linestyle=line_style, linewidth=1, label="Final model (epoch 245)")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision Across Model Training", fontsize=16)
plt.xlim(df["epoch"].min(), df["epoch"].max())
plt.ylim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower left', frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/FigA_precision.png", dpi=dpi)
plt.close()

# =========================
# Panel B: Recall vs Epoch
# =========================
plt.figure(figsize=(5, 4))
plt.plot(epochs, df["metrics/recall(B)"], label="Recall", linewidth=1.5)
plt.axvline(x=best_epoch, color=line_color, linestyle=line_style, linewidth=1, label="Final model (epoch 245)")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Recall", fontsize=14)
plt.title("Recall Across Model Training", fontsize=16)
plt.xlim(df["epoch"].min(), df["epoch"].max())
plt.ylim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower left', frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/FigB_recall.png", dpi=dpi)
plt.close()

# =========================
# Panel C: mAP@50–95 vs Epoch
# =========================
plt.figure(figsize=(5, 4))
plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@50–95", linewidth=1.5)
plt.axvline(x=best_epoch, color=line_color, linestyle=line_style, linewidth=1, label="Final model (epoch 245)")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("mAP@50–95", fontsize=14)
plt.title("Mean Average Precision During Training", fontsize=16)
plt.xlim(df["epoch"].min(), df["epoch"].max())
plt.ylim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower left', frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/FigC_mAP50_95.png", dpi=dpi)
plt.close()

# =========================
# Panel D: Validation DFL Loss vs Epoch
# =========================
plt.figure(figsize=(5, 4))
plt.plot(epochs, df["val/dfl_loss"], label="Validation DFL Loss", linewidth=1.5)
plt.axvline(x=best_epoch, color=line_color, linestyle=line_style, linewidth=1, label="Final model (epoch 245)")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("DFL Loss", fontsize=14)
plt.title("Validation Localization Uncertainty", fontsize=16)
plt.xlim(df["epoch"].min(), df["epoch"].max())
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper right', frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/FigD_val_DFL_loss.png", dpi=dpi)
plt.close()

print(f"All figures saved in folder: {output_dir}")
