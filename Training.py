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

for max_depth_hyperparameter in range(5, 15, 5):
    for n_estimators_hyperparameter in range(50, 150, 50):
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
                                "Training Accuracy": round(accuracy_against_traning_data * 100, 2),
                                "Test Accuracy": round(accuracy_against_test_data * 100, 2)})

        print(f"Training using: Max Depth={max_depth_hyperparameter}, N Estimators={n_estimators_hyperparameter}")

#Collecting the results in a dataframe and saving to a csv file
training_metrics = pd.DataFrame(training_output, columns=["Max Depth","N Estimators","Training Accuracy","Test Accuracy"])
training_metrics.to_csv("training_metrics.csv", index=False)

#Visualizing the results
plt.subplot(2, 1, 1)
plt.title("Hyperparameter Tuning accuracy results")
plt.barh(training_metrics["Max Depth"].unique(), training_metrics["Test Accuracy"].groupby(training_metrics["Max Depth"]).mean())
plt.ylabel("Max Depth")

plt.subplot(2, 1, 2)
plt.barh(training_metrics["N Estimators"].unique(), training_metrics["Test Accuracy"].groupby(training_metrics["N Estimators"]).mean())
plt.xlabel("Average Accuracy (%)")
plt.ylabel("N Estimators")

plt.savefig("hyperparameter_tuning_results.png")
plt.show()