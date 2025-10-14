import pandas as pd

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import LinearSVC as svc

training_data = pd.read_csv("wildfires_training.csv")
testing_data = pd.read_csv("wildfires_test.csv")

training_parameters = training_data.drop(columns=["fire"])
training_target = training_data["fire"]

testing_parameters = testing_data.drop(columns=["fire"])
testing_target = testing_data["fire"]

classifier = rf(n_estimators=50,max_depth=5)
classifier.fit(training_parameters,training_target)

