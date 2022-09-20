#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from gama.configuration.fasttextclassifier import FastTextClassifier

def remove_non_string_cols(X):
  # Only keep the columns with string values
  print("Let's ditch the non-string columns broooo")
  return X[[col_name for col_name in X.columns if X[col_name].dtype == np.dtype('O')]]

if __name__ == "__main__":
  # Getting the data set

  #X, y = load_breast_cancer(return_X_y=True)
  # dataset = openml.datasets.get_dataset(42078) # Beer reviews
  # dataset = openml.datasets.get_dataset(42803) # Road safety
  dataset = openml.datasets.get_dataset(42132) # Traffic violations
  X, y, _, _ = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
  )

  X = remove_non_string_cols(X)

  #y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
  if isinstance(y, np.ndarray):
    y = pd.Series(LabelEncoder().fit_transform(y))
  else:
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  clf = FastTextClassifier()
  pre_fit = time.time()
  clf.fit(X_train, y_train)
  training_time = time.time() - pre_fit
  print(f"training time: {training_time:.2f}")
  score = cross_val_score(clf, X_train, y_train)
  print("CV scores:", score)
  print("average CV score:", sum(score) / len(score))
