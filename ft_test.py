#!/usr/bin/env python3
import time
import os
import numpy as np
import pandas as pd
import openml
import json
import pickle
from tqdm import trange
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
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder 

def dummy_test(X, y, cv):
  # Only keep the columns with string values
  # X = X[[col_name for col_name in X.columns if X[col_name].dtype == np.dtype('O')]]

  if isinstance(y, np.ndarray):
    y = pd.Series(LabelEncoder().fit_transform(y))
  else:
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

  clf = make_pipeline(OneHotEncoder(handle_unknown="ignore"), SimpleImputer(strategy="most_frequent"), RandomForestClassifier())

  if len(np.unique(y)) < 2:
    raise ValueError("y_true has fewer than 2 unique values.")
  if len(np.unique(y)) == 2:
    scorer = "roc_auc"
  else:
    scorer = "neg_log_loss"

  scores = cross_val_score(
    clf,
    X,
    y, 
    cv=cv,
    scoring=scorer, 
    n_jobs=-1
  )

  return list(scores)

def fasttext_run(X, y, cv, pretrainedVectors="", dim=100):
  # Only keep the columns with string values
  # X = X[[col_name for col_name in X.columns if X[col_name].dtype == np.dtype('O')]]

  if isinstance(y, np.ndarray):
    y = pd.Series(LabelEncoder().fit_transform(y))
  else:
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

  clf = FastTextClassifier(minn=0, maxn=0, epoch=5, lr=0.1, pretrainedVectors=pretrainedVectors, dim=dim, autotune=True)

  if len(np.unique(y)) < 2:
    raise ValueError("y_true has fewer than 2 unique values.")
  if len(np.unique(y)) == 2:
    scorer = "roc_auc"
  else:
    scorer = "neg_log_loss"

  scores = cross_val_score(
    clf,
    X,
    y, 
    cv=cv,
    scoring=scorer, 
    n_jobs=-1
  )
  
  # fill the nannies 
  if any(np.isnan(scores)):
    dum_score = cross_val_score(DummyClassifier(), X, y, cv=cv, scoring=scorer)
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
  fol_name = "openml-cc18-ft-autotuned"
  if not os.path.isdir(fol_name):
    os.mkdir(fol_name)
  with open(f"{fol_name}/results_{dataset_scores['data_id']}_{int(time.time())}.json", "w+") as f:
    json.dump(dataset_scores, f)

def main(t_ids):
  for i, t_id in zip(trange(len(t_ids)), t_ids):
    dataset_scores = {}
    try:
      task = openml.tasks.get_task(t_id)
      dataset = task.get_dataset(d_id)
      X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
      )
      
      dataset_scores["data_id"] = dataset.id
      dataset_scores["name"] = dataset.name
      dataset_scores["metric"] = "roc_auc" if len(np.unique(y)) == 2 else "neg_log_loss"

      # retrieve the predefined cv folds for experimentation
      with open(f"cv_folds/{dataset.id}.pkl", "rb") as f:
        cv = pickle.load(f)

    except Exception as e:
      print(f"{t_id} OpenML and data prep. failed: {e}\n")
      log_error(e)
      log_score(dataset_scores)
      continue

#    try:
#      dataset_scores["fasttext"] = fasttext_run(X, y, cv=cv)
#    except Exception as e:
#      print(f"{t_id} fasttext failed: {e}\n")
#      log_error(e)
#
#    try:
#      dataset_scores["fasttext_100"] = fasttext_run(X, y, cv=cv, pretrainedVectors="100.vec", dim=100)
#    except Exception as e:
#      print(f"{t_id} fasttext PT failed: {e}\n")
#      log_error(e)

    try:
      dataset_scores["fasttext_300"] = fasttext_run(X, y, cv=cv, pretrainedVectors="300.vec", dim=300)
    except Exception as e:
      print(f"{t_id} fasttext PT failed: {e}\n")
      log_error(e)

#    try:
#      dataset_scores["random_forest"] = dummy_test(X, y, cv=cv)
#    except Exception as e:
#      print(f"{t_id} fasttext failed: {e}\n")
#      log_error(e)
#
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
  benchmark_suite = openml.study.get_suite('OpenML-CC18')  # obtain the benchmark suite

  for _ in range(1):
    main(benchmark_suite.tasks)

