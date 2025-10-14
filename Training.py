import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import LinearSVC as svc
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

training_data = pd.read_csv("wildfires_training.csv")
testing_data = pd.read_csv("wildfires_test.csv")

training_parameters = training_data.drop(columns=["fire"])
training_target = training_data["fire"]

testing_parameters = testing_data.drop(columns=["fire"])
testing_target = testing_data["fire"]

#Defing hyperparameters for tuning
max_depth_hyperparameter = 10
n_estimators_hyperparameter = 10

#Seeting up a list to collect the training results
training_output = []

for max_depth_hyperparameter in range(5, 105, 5):
    for n_estimators_hyperparameter in range(50, 2050, 50):
        #Traning the Random Forest Classifier using the varied hyperparameters
        classifier = rf(max_depth=max_depth_hyperparameter,n_estimators=n_estimators_hyperparameter)
        classifier.fit(training_parameters,training_target)

        prediction_on_training_data = classifier.predict(training_parameters)
        prediction_on_test_data = classifier.predict(testing_parameters)

        #Evaluating the model performance
        accuracy_against_traning_data = accuracy_score(training_target,prediction_on_training_data)
        accuracy_against_test_data = accuracy_score(testing_target,prediction_on_test_data)

        #Appending the results to a list
        training_output.append({"Max Depth": max_depth_hyperparameter,
                                "N Estimators": n_estimators_hyperparameter,
                                "Training Accuracy": int(round(accuracy_against_traning_data * 100, 0)),
                                "Test Accuracy":  int(round(accuracy_against_test_data * 100, 0))})

        print(f"Training using: Max Depth={max_depth_hyperparameter}, N Estimators={n_estimators_hyperparameter}")

#Collecting the results in a dataframe and saving to a csv file
training_metrics = pd.DataFrame(training_output, columns=["Max Depth","N Estimators","Training Accuracy","Test Accuracy"])
training_metrics.to_csv("training_metrics.csv", index=False)

#Visualizing the results
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