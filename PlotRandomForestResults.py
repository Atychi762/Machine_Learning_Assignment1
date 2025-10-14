import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

training_metrics = pd.read_csv("datasets/training_metrics.csv")


filtered_metrics = training_metrics[training_metrics["N Estimators"] == 1000]
filtered_metrics = filtered_metrics.sort_values("Max Depth")

plt.subplot(2, 1, 1)
plt.title("Test Accuracy when varying Max Depth (N Estimators=1000)")
plt.plot(filtered_metrics["Max Depth"], filtered_metrics["Test Accuracy"], marker='.') 
plt.ylabel("Accuracy (%)")
plt.xlabel("Max Depth")
plt.tight_layout(pad=2)

filtered_metrics = training_metrics[training_metrics["Max Depth"] == 50]
filtered_metrics = filtered_metrics.sort_values("N Estimators")

plt.subplot(2, 1, 2)
plt.title("Test Accuracy when varying N Estimators (Max Depth=50)")
plt.plot(filtered_metrics["N Estimators"], filtered_metrics["Test Accuracy"], marker='.') 
plt.ylabel("Accuracy (%)")
plt.xlabel("N Estimators")
plt.tight_layout(pad=2)

plt.savefig("hyperparameter_tuning_results.png")
plt.show()

pivot = training_metrics.pivot_table(
    index="Max Depth",
    columns="N Estimators",
    values="Test Accuracy",
    aggfunc="mean")

pivot = pivot.sort_index().sort_index(axis=1)
n_rows, n_cols = pivot.shape
# heuristic for annotation font size
annot_size = max(6, min(14, int(300 / (max(1, n_rows * n_cols) ** 0.5))))

fig_w = max(10, 0.35 * n_cols)
fig_h = max(8, 0.35 * n_rows)
plt.figure(figsize=(fig_w, fig_h))

sns.heatmap(
    pivot,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    cbar_kws={"label": "Test Accuracy (%)"},
    linewidths=0.5,
    annot_kws={"size": annot_size}
)

plt.xlabel("N Estimators")
plt.ylabel("Max Depth")
plt.title("Test Accuracy across Hyperparameters")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("hyperparameter_accuracy_grid.png", bbox_inches="tight", dpi=150)
plt.show()