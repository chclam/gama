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
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, log_loss
from gama.postprocessing.ensemble import EnsemblePostProcessing
from gama.configuration.fasttextclassifier import FastTextClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def random_forest(X, y, cv):
  if isinstance(y, np.ndarray):
    y = pd.Series(LabelEncoder().fit_transform(y))
  else:
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

  str_cols = [col for col in X.columns if isinstance(X[col].dtype, pd.core.dtypes.dtypes.CategoricalDtype) or X[col].dtype in ["O", '<M8[ns]']]
  high_card_cols = [col for col in str_cols if len(X[col].unique()) > 30]
  low_card_cols = [col for col in str_cols if len(X[col].unique()) <= 30]

  clf = make_pipeline(
    make_column_transformer(
      (OneHotEncoder(handle_unknown="ignore"), low_card_cols),
      (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), high_card_cols),
      remainder="passthrough"
    ),
    SimpleImputer(strategy="most_frequent"),
    RandomForestClassifier()
  )

  if len(np.unique(y)) < 2:
    raise ValueError("y_true has fewer than 2 unique values.")
  if len(np.unique(y)) == 2:
    scorer = "roc_auc"
    score_func = roc_auc_score
  else:
    scorer = "neg_log_loss"
    score_func = log_loss

  scores = cross_val_score(
    clf,
    X,
    y, 
    cv=cv,
    #scoring=scorer, 
    scoring=make_scorer(score_func, greater_is_better=(scorer=="roc_auc"), needs_proba=True, labels=y.unique()),
    n_jobs=3,
    #error_score="raise"
  )

  return list(scores)

def fasttext_run(X, y, cv, pretrainedVectors="", dim=100, autotune=False, thread=None):
  # Only keep the columns with string values
  # X = X[[col_name for col_name in X.columns if X[col_name].dtype == np.dtype('O')]]

  if isinstance(y, np.ndarray):
    y = pd.Series(LabelEncoder().fit_transform(y))
  else:
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

  if autotune:
    clf = FastTextClassifier(autotune=autotune, thread=thread)
  else:
    clf = FastTextClassifier(minn=0, maxn=0, epoch=5, lr=0.1, pretrainedVectors=pretrainedVectors, dim=dim, autotune=autotune)

  if len(np.unique(y)) < 2:
    raise ValueError("y_true has fewer than 2 unique values.")
  if len(np.unique(y)) == 2:
    scorer = "roc_auc"
    score_func = roc_auc_score
  else:
    scorer = "neg_log_loss"
    score_func = log_loss
  
  from sklearn.metrics import make_scorer

  import time
  start = time.time()
  scores = cross_val_score(
    clf,
    X,
    y,
    cv=cv,
    #scoring=scorer,
    scoring=make_scorer(score_func, greater_is_better=(scorer == "roc_auc"), needs_proba=True, labels=y.unique()),
    n_jobs=2 if autotune else -1,
    error_score="raise",
  )

#  import time
#  start = time.time()
#  from sklearn.model_selection import RandomizedSearchCV
#  params = {
#    "lr": np.arange(0, 1.1, 0.1),
#    "epoch": [15],
#    #"wordNgrams": np.arange(0, 3, 1),
#    #"maxn": [0, 1, 2, 3]
#  }
#
#  search = RandomizedSearchCV(
#    clf,
#    params,
#    cv=cv,
##    scoring=scorer, 
#    scoring=make_scorer(score_func, greater_is_better=(scorer == "roc_auc"), needs_proba=True, labels=y.unique()),
#    n_jobs=1
#  ).fit(X, y)
#
#  best_index = np.argmin(search.cv_results_["rank_test_score"])
#  scores = [search.cv_results_[f'split{k}_test_score'][best_index] for k in range(0, 5)]
#
  train_time = time.time() - start
  
  print(f"{scorer}:", scores)
  # fill the nannies 
  if any(np.isnan(scores)):
    dum_score = cross_val_score(DummyClassifier(), X, y, cv=cv, scoring=scorer)
    scores = np.nan_to_num(scores, nan=np.nanmean(dum_score))

  if os.path.isdir("cache/"):
    os.system("rm -rf cache/")
  return {"scores": list(scores), "time": train_time}

def log_error(e):
  with open("script_error.txt", "a+") as out:
    out.write(str(e))
  print(e)

def log_score(dataset_scores, setup_name):
  fol_name = f"custom-dataset-{setup_name}"
  #fol_name = "custom-dataset-ft"
  if not os.path.isdir(fol_name):
    os.mkdir(fol_name)
  with open(f"{fol_name}/results_{dataset_scores['data_id']}_{int(time.time())}.json", "w+") as f:
    json.dump(dataset_scores, f)


def load_dataset_from_disk(d_id, path="."):
  try:
    with open(f"{path}/{d_id}_X.pkl", "rb") as f:
      X = pickle.load(f)
    with open(f"{path}/{d_id}_y.pkl", "rb") as f:
      y = pickle.load(f)
  except Exception as e:
    raise f"Error retrieving dataset from disk: {e}"
  return X, y

def main(ids, suite):
  setups = {
    #"ft": {"pretrainedVectors": "", "dim": 100, "autotune": False},
    #"ft-100": {"pretrainedVectors": "100.vec", "dim": 100, "autotune": False},
    #"ft-300": {"pretrainedVectors": "300.vec", "dim": 300, "autotune": False},
    #"ft-random-cv": {"pretrainedVectors": "", "dim": 100, "autotune": False},
    #"ft-autotune": {"pretrainedVectors": "", "dim": 100, "autotune": True, "thread": 1},
    "random-forest": None,
  }

  for setup_name, setup in setups.items():
    #for i, id_ in zip(trange(len(ids)), ids):
    for i, (dataname, id_) in zip(trange(len(ids)), ids):
      dataset_scores = {}
      try:
        if suite == "custom":
          X, y = load_dataset_from_disk(id_, "prepped_data")
        elif suite == "openml_cc18":
          task = openml.tasks.get_task(id_)
          dataset = task.get_dataset()
          X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
          )
        else:
          raise ValueError("Invalid value for suite.")
        
        dataset_scores["data_id"] = id_
        dataset_scores["name"] = dataname
        #dataset_scores["data_id"] = dataset.id
        #dataset_scores["name"] = dataset.name
        dataset_scores["metric"] = "roc_auc" if len(np.unique(y)) == 2 else "neg_log_loss"

        # retrieve the predefined cv folds for experimentation
        with open(f"cv_folds/{id_}.pkl", "rb") as f:
          cv = pickle.load(f)

      except Exception as e:
        print(f"{id_} OpenML and data prep. failed: {e}\n")
        log_error(e)
        log_score(dataset_scores, setup_name)
        continue

#      try:
#        dataset_scores[f"fasttext_{setup_name}_time"] = fasttext_run(X, y, cv=cv, **setup)
#      except Exception as e:
#        import traceback
#        traceback.print_exc()
#        print(f"{id_} fasttext PT failed: {e}\n")
#        log_error(e)
  
  #    try:
  #      dataset_scores["fasttext_time"] = fasttext_run(X, y, cv=cv, pretrainedVectors="100.vec", dim=100)
  #    except Exception as e:
  #      print(f"{id_} fasttext PT failed: {e}\n")
  #      log_error(e)

  #    try:
  #      dataset_scores["fasttext_300_time"] = fasttext_run(X, y, cv=cv, pretrainedVectors="300.vec", dim=300)
  #    except Exception as e:
  #      print(f"{id_} fasttext PT failed: {e}\n")
  #      log_error(e)
  #
  #    try:
  #      dataset_scores["fasttext_auto_time"] = fasttext_run(X, y, cv=cv, autotune=True)
  #    except Exception as e:
  #      print(f"{id_} fasttext PT failed: {e}\n")
  #      log_error(e)

  #    try:
  #      dataset_scores["fasttext_random_search"] = fasttext_run(X, y, cv=cv)
  #    except Exception as e:
  #      import traceback
  #      traceback.print_exc()
  #      print(f"{id_} fasttext PT failed: {e}\n")
  #      log_error(e)
  #
      try:
        dataset_scores["random_forest"] = random_forest(X, y, cv=cv)
      except Exception as e:
        print(f"{id_} fasttext failed: {e}\n")
        log_error(e)

      try:
        log_score(dataset_scores, setup_name)
      except Exception as e:
        print(e)
    

if __name__ == "__main__":
  # Getting the data set

  ids = {
#    "beerreviews": 42078,
#    "road_safety": 42803,
#    "traffic_violations": 42132,
#    "drug_directory": 43044,
#    "kickstarter": 42076,
#    "openpayments": 42738,
#    "midwest": 42530,
    "metobjects": "MET_OBJECTS",
    "cacao_flavors": 42133,
    "anime_data": 43508,
    "crime_data": 43579,
    "colleges": 42723,
    "wine_review": 43651,
    "public_procurement": 42163,
    "federal_elections": 42080,
  }

  openml.config.apikey = "6582f698e8e48968a7566b87fff8a75e"  # set the OpenML Api Key
  benchmark_suite = openml.study.get_suite('OpenML-CC18')  # obtain the benchmark suite

  for _ in range(1):
    main(ids.items(), "custom") # custom dataset
    #main(benchmark_suite.tasks, "openml_cc18") # openml-cc18

