import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

training_metrics = pd.read_csv("datasets/linearSVC_training_metrics.csv")

plt.scatter(training_metrics["C value"], training_metrics["Test Accuracy"], c=training_metrics["Num Iterations"], cmap='viridis', marker='o')
plt.colorbar(label='Num Iterations')
plt.xscale('log')
plt.xlabel('C Value (log scale)')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs C Value with Num Iterations as Color')
plt.grid(True)
plt.savefig("linearSVC_hyperparameter_tuning_results.png")
plt.show()