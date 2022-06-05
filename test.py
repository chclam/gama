#!/usr/bin/env python3
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from gama import GamaClassifier
import os
import numpy as np
import openml
import pandas as pd

if __name__ == '__main__':
  CACHE_FOLDER = "cache_data"

  if "X.pkl" in os.listdir(CACHE_FOLDER) and "y.pkl" in os.listdir(CACHE_FOLDER):
    X = pd.read_pickle(f"{CACHE_FOLDER}/X.pkl")
    y = pd.read_pickle(f"{CACHE_FOLDER}/y.pkl")
  else:
    dataset = openml.datasets.get_dataset(42078) # Beer review data set
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute,
                                  dataset_format="dataframe")
    X.to_pickle(f"{CACHE_FOLDER}/X.pkl")
    y.to_pickle(f"{CACHE_FOLDER}/y.pkl")

  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

  automl = GamaClassifier(max_total_time=10, store="nothing")
  print("Starting fit.")
  automl.fit(X_train, y_train)

  label_predictions = automl.predict(X_test)
  probability_predictions = automl.predict_proba(X_test)

  print('accuracy:', accuracy_score(y_test, label_predictions))
  print('log loss:', log_loss(y_test, probability_predictions))
  print('log_loss', automl.score(X_test, y_test))
