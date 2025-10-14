import pandas as pd

from sklearn.svm import LinearSVC as svc
from sklearn.metrics import accuracy_score


training_data = pd.read_csv("datasets/wildfires_training.csv")
testing_data = pd.read_csv("datasets/wildfires_test.csv")

training_parameters = training_data.drop(columns=["fire"])
training_target = training_data["fire"]

testing_parameters = testing_data.drop(columns=["fire"])
testing_target = testing_data["fire"]
