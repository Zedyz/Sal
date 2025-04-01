import math
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_runs/training_log_convnext.csv")

metrics = ["loss", "nss", "cc", "kl", "aucj", "sauc"]
epochs = df["epoch"]

num_metrics = len(metrics)
ncols = 3
nrows = math.ceil(num_metrics / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    train_col = f"train_{metric}"
    val_col   = f"val_{metric}"

    ax.plot(epochs, df[train_col], label=f"Train {metric}")
    ax.plot(epochs, df[val_col],   label=f"Val {metric}")

    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()}")
    ax.legend()

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Training vs. Validation Metrics Over Epochs", fontsize=16)
plt.tight_layout(pad=3.0)
plt.show()

