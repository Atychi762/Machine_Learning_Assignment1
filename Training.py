import pandas as pd

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import LinearSVC as svc

training_data = pd.read_csv("wildfires_training.csv")


