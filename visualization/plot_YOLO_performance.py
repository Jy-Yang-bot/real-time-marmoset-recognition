"""
Plot the learning curve of the YOLO model, throughout the training iterations/epochs.
- Input: .csv file generated with the YOLO training.
- Output: figures of learning curve.
- A .csv file is generated after the training of each YOLO model is completed.
- Multiple performance evaluation metrics are included in this .csv file, thus we can plot the learning curve from this file.
- Example metrics are: precision, recall, mAP@50-95, DFL loss

Contributor: Jiayue Yang, 2025-08-22
"""
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

# define parameters
csv_path = "results.csv" # (to edit) the directory of the result .csv file of the YOLO training
output_dir = "figures" # (to edit) the directory where the output learning curve figures will be saved
dpi = 300 # (to edit) figure resolution
best_epoch = n # (to edit) the epoch of best model selection --> will be known at the end of the training
line_color = "darkred" # (to edit) vertical line color, can choose based on preference
line_style = "--" # (to edit)

# assess and load the csv
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(csv_path)
epochs = df["epoch"]

# learning curve plotting
# (to edit) figure size, font size, axis labels, and the titles
# precision
plt.figure(figsize=(5, 4))
plt.plot(epochs, df["metrics/precision(B)"], label="Precision", linewidth=1.5)
# draw a vertical line where the best epoch is on the learning curve
plt.axvline(x=best_epoch, color=line_color, linestyle=line_style, linewidth=1, label="Final model (best epoch)")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision Across Model Training", fontsize=16)
plt.xlim(df["epoch"].min(), df["epoch"].max())
plt.ylim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower left', frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/Figure1_precision.png", dpi=dpi)
plt.close()

# recall
plt.figure(figsize=(5, 4))
plt.plot(epochs, df["metrics/recall(B)"], label="Recall", linewidth=1.5)
# draw a vertical line where the best epoch is on the learning curve
plt.axvline(x=best_epoch, color=line_color, linestyle=line_style, linewidth=1, label="Final model (best epoch)")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Recall", fontsize=14)
plt.title("Recall Across Model Training", fontsize=16)
plt.xlim(df["epoch"].min(), df["epoch"].max())
plt.ylim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower left', frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/Figure2_recall.png", dpi=dpi)
plt.close()

# mAP@50-95
plt.figure(figsize=(5, 4))
plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@50–95", linewidth=1.5)
# draw a vertical line where the best epoch is on the learning curve
plt.axvline(x=best_epoch, color=line_color, linestyle=line_style, linewidth=1, label="Final model (best epoch)")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("mAP@50–95", fontsize=14)
plt.title("Mean Average Precision During Training", fontsize=16)
plt.xlim(df["epoch"].min(), df["epoch"].max())
plt.ylim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower left', frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/Figure3_mAP50_95.png", dpi=dpi)
plt.close()

# DFL loss
plt.figure(figsize=(5, 4))
plt.plot(epochs, df["val/dfl_loss"], label="Validation DFL Loss", linewidth=1.5)
# draw a vertical line where the best epoch is on the learning curve
plt.axvline(x=best_epoch, color=line_color, linestyle=line_style, linewidth=1, label="Final model (best epoch)")
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

# if all figures are done, print the ending statement
print(f"All figures are complated, results saved in: {output_dir}")

