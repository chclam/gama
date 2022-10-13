#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from gama.configuration.fasttextclassifier import FastTextClassifier
from gama import GamaClassifier
from gama.search_methods import AsynchronousSuccessiveHalving
from sklearn.pipeline import make_pipeline 


if __name__ == "__main__":
  # Getting the data set

  #X, y = load_breast_cancer(return_X_y=True)
  # dataset = openml.datasets.get_dataset(42078) # Beer reviews
  # dataset = openml.datasets.get_dataset(42803) # Road safety
  dataset = openml.datasets.get_dataset(42132) # Traffic violations
  X, y, _, _ = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
  )

  X = X[:100000]
  y = y[:100000]

  #y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
  if isinstance(y, np.ndarray):
    y = pd.Series(LabelEncoder().fit_transform(y))
  else:
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  automl = GamaClassifier(max_total_time=600, store="logs", max_eval_time=450, scoring="accuracy")
  print("Starting `fit` which will take roughly 3 minutes.")
  automl.fit(X_train, y_train)

  label_predictions = automl.predict(X_test)
  probability_predictions = automl.predict_proba(X_test)
  
  print('accuracy:', accuracy_score(y_test, label_predictions))
  print('log loss:', log_loss(y_test, probability_predictions))
  # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
  print('log_loss', automl.score(X_test, y_test))
