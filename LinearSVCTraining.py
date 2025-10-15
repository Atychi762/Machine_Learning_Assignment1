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
max_iteration_hyperparameter = 1000
c_hyperparameter = 0.0001

scaler = StandardScaler()
scaled_training_parameters = scaler.fit_transform(training_parameters)
scaled_testing_parameters = scaler.transform(testing_parameters)

classifier = svc(C=1, max_iter=max_iteration_hyperparameter)
classifier.fit(scaled_training_parameters,training_target)

prediction_on_training_data = classifier.predict(scaled_training_parameters)
prediction_on_test_data = classifier.predict(scaled_testing_parameters)

#Evaluating the model performance
accuracy_against_traning_data = accuracy_score(training_target,prediction_on_training_data)
accuracy_against_test_data = accuracy_score(testing_target,prediction_on_test_data)

print(f"Training accuracy: {accuracy_against_traning_data: .4f}")
print(f"Testing accuracy: {accuracy_against_test_data: .4f}")