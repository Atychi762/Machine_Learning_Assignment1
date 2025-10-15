import pandas as pd

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import accuracy_score

training_data = pd.read_csv("datasets/wildfires_training.csv")
testing_data = pd.read_csv("datasets/wildfires_test.csv")

training_parameters = training_data.drop(columns=["fire"])
training_target = training_data["fire"]

testing_parameters = testing_data.drop(columns=["fire"])
testing_target = testing_data["fire"]

#Defing hyperparameters for tuning
max_depth_hyperparameter = 10
n_estimators_hyperparameter = 10

#Seting up a list to collect the training results
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
training_metrics.to_csv("datasets/random_forest_raining_metrics.csv", index=False)