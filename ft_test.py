#!/usr/bin/env python3
import time
import os
import numpy as np
import pandas as pd
import openml
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve
from gama.postprocessing.ensemble import EnsemblePostProcessing
from gama.configuration.fasttextclassifier import FastTextClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

def fasttext_run(X, y, cv, pretrainedVectors="", dim=100):
  # Only keep the columns with string values
  # X = X[[col_name for col_name in X.columns if X[col_name].dtype == np.dtype('O')]]

  if isinstance(y, np.ndarray):
    y = pd.Series(LabelEncoder().fit_transform(y))
  else:
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

  clf = FastTextClassifier(minn=0, maxn=0, epoch=5, lr=0.1, pretrainedVectors=pretrainedVectors, dim=dim)

  if len(np.unique(y)) < 2:
    raise ValueError("y_true has fewer than 2 unique values.")
  if len(np.unique(y)) == 2:
    scorer = "roc_auc"
  else:
    scorer = "neg_log_loss"

  scores = cross_val_score(clf,
    X,
    y, 
    cv=cv,
    scoring=scorer, 
    n_jobs=-1
  )
  
  # fill the nannies 
  if any(np.isnan(scores)):
    dum_score = cross_val_score(DummyClassifier(), X, y, cv=5, scoring=scorer)
    scores = np.nan_to_num(scores, nan=np.nanmean(dum_score))

  print(f"{scorer}:", scores)
  if os.path.isdir("cache/"):
    os.system("rm -rf cache/")
  return list(scores)

def log_error(e):
  with open("script_error.txt", "a+") as out:
    out.write(str(e))
  print(e)

def log_score(dataset_scores):
  if not os.path.isdir(f"roc_log_results"):
    os.mkdir("roc_log_results")
  with open(f"roc_log_results/results_{dataset_scores['data_id']}_{int(time.time())}.json", "w+") as f:
    json.dump(dataset_scores, f)

def main(ids):
  #for ds_name, d_id in ids.items():
  for d_id in ids:
      
    dataset_scores = {
      "data_id": d_id,
      #"data_name": ds_name 
    }

    try:
      dataset = openml.datasets.get_dataset(d_id)
      X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
      )

      if d_id == 42076:
        y = X["state"]
        del X["state"]

      # Cap to 100.000 instances
      if len(X) > 100000:
        X, _, y, _ = train_test_split(X, y, stratify=y, train_size=100000, random_state=0, shuffle=True)

      dataset_scores["name"] = dataset.name
      dataset_scores["metric"] = "roc_auc" if len(np.unique(y)) == 2 else "neg_log_loss"

    except Exception as e:
      print(f"{d_id} openml failed: {e}\n")
      log_error(e)
      log_score(dataset_scores)
      continue

    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

    try:
      dataset_scores["fasttext"] = fasttext_run(X, y, cv=cv.split(X, y))
    except Exception as e:
      print(f"{d_id} fasttext failed: {e}\n")
      log_error(e)

    try:
      dataset_scores["fasttext_100"] = fasttext_run(X, y, cv=cv.split(X, y), pretrainedVectors="100.vec", dim=100)
    except Exception as e:
      print(f"{d_id} fasttext PT failed: {e}\n")
      log_error(e)

    try:
      log_score(dataset_scores)
    except Exception as e:
      print(e)

if __name__ == "__main__":
  # Getting the data set

#  ids = {
#    "beerreviews": 42078,
#    "road_safety": 42803,
#    "traffic_violations": 42132,
#    "drug_directory": 43044,
#    "kickstarter": 42076,
#    "openpayments": 42738,
#  }

  openml.config.apikey = "6582f698e8e48968a7566b87fff8a75e"  # set the OpenML Api Key
#  benchmark_suite = openml.study.get_suite('OpenML-CC18')  # obtain the benchmark suite

  ids = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 43, 49, 53, 3560, 3902, 3904, 3913, 3917, 3918, 42076, 42078, 42132, 42738, 42803, 43044]

  for _ in range(1):
    main(ids)

