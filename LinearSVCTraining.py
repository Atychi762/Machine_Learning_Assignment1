import pandas as pd

from sklearn.svm import LinearSVC as svc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#Loading the training and testing datasets
training_data = pd.read_csv("datasets/wildfires_training.csv")
testing_data = pd.read_csv("datasets/wildfires_test.csv")

training_parameters = training_data.drop(columns=["fire"])
training_target = training_data["fire"]

testing_parameters = testing_data.drop(columns=["fire"])
testing_target = testing_data["fire"]


#Defing hyperparameters for tuning
max_iteration_hyperparameter = [1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000]
c_hyperparameter = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000] #Iterating C in a logarithmic scale

#Scaling the dataset 
scaler = StandardScaler()
scaled_training_parameters = scaler.fit_transform(training_parameters)
scaled_testing_parameters = scaler.transform(testing_parameters)

#Seting up a list to collect the training results
training_output = []

for c in c_hyperparameter:
    for max_iterations in max_iteration_hyperparameter:
       #Training the model
        classifier = svc(C=c, max_iter=max_iterations)
        classifier.fit(scaled_training_parameters,training_target)

        #Make predictions
        prediction_on_training_data = classifier.predict(scaled_training_parameters)
        prediction_on_test_data = classifier.predict(scaled_testing_parameters)

        #Evaluating the model performance
        accuracy_against_traning_data = accuracy_score(training_target,prediction_on_training_data)
        accuracy_against_test_data = accuracy_score(testing_target,prediction_on_test_data)

       #Appending the results to a list
        training_output.append({"C value": c,
                                "Num Iterations": max_iterations,
                                "Training Accuracy": round(accuracy_against_traning_data * 100, 2),
                                "Test Accuracy":  round(accuracy_against_test_data * 100, 2)})

        print(f"Training using: C value={c}, Num Iterations={max_iterations}")

#Collecting the results in a dataframe and saving to a csv file
training_metrics = pd.DataFrame(training_output, columns=["C value","Num Iterations","Training Accuracy","Test Accuracy"])
training_metrics.to_csv("datasets/linearSVC_training_metrics.csv", index=False)