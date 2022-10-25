#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
from gama.configuration.fasttextclassifier import FastTextClassifier
from gama import GamaClassifier
from gama.search_methods import AsynchronousSuccessiveHalving
from gama.postprocessing import EnsemblePostProcessing
from sklearn.pipeline import make_pipeline 
from imblearn.over_sampling import RandomOverSampler


if __name__ == "__main__":
  # Getting the data set

  #X, y = load_breast_cancer(return_X_y=True)
  dataset = openml.datasets.get_dataset(42078) # Beer reviews
  # dataset = openml.datasets.get_dataset(42803) # Road safety
  #dataset = openml.datasets.get_dataset(42132) # Traffic violations
  X, y, _, _ = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
  )

  #y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
  if isinstance(y, np.ndarray):
    y = pd.Series(LabelEncoder().fit_transform(y), index=X.index)
  else:
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)


  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=150000, test_size=20000)

#  if min(X_train.value_counts()) < 10:
#    ros = RandomOverSampler(random_state=0, sampling_strategy="minority")
#    X_train, y_train = ros.fit_resample(X_train, y_train)


  automl = GamaClassifier(max_total_time=700, store="logs", max_eval_time=600, scoring="roc_auc_ovo", post_processing=EnsemblePostProcessing(), search=AsynchronousSuccessiveHalving())
  #automl = GamaClassifier(max_total_time=600, store="logs", max_eval_time=450, scoring="accuracy")
  print("Starting `fit` which will take roughly 3 minutes.")
  automl.fit(X_train, y_train)

  label_predictions = automl.predict(X_test)
  probability_predictions = automl.predict_proba(X_test)
  import pdb; pdb.set_trace()
  
  print('accuracy:', accuracy_score(y_test, label_predictions))
  print('f1-score:', f1_score(y_test, label_predictions, average="macro"))
  print('roc_auc:', roc_auc_score(y_test, probability_predictions, average="macro", multi_class="ovr"))
  print('log loss:', log_loss(y_test, probability_predictions))
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  import matplotlib.pyplot as plt
  cm = confusion_matrix(y_test, label_predictions, labels=y.unique())
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.unique())
  disp.plot(include_values=False)
  plt.show()

  # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
  #print('log_loss', automl.score(X_test, y_test))
