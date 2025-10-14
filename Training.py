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
max_depth_hyperparameter = 50
n_estimators_hyperparameter = 3000

#Traning the Random Forest Classifier
classifier = rf(max_depth=max_depth_hyperparameter,n_estimators=n_estimators_hyperparameter)
classifier.fit(training_parameters,training_target)

prediction_on_training_data = classifier.predict(training_parameters)
prediction_on_test_data = classifier.predict(testing_parameters)

#Evaluating the model performance
accuracy_against_traning_data = accuracy_score(training_target,prediction_on_training_data)
accuracy_against_test_data = accuracy_score(testing_target,prediction_on_test_data)
#Printing the accuracies for the moment(need to write to file later)
print(f"Accuracy against training data: {accuracy_against_traning_data * 100:.2f}%")
print(f"Accuracy against test data: {accuracy_against_test_data * 100:.2f}%")


