#!/usr/bin/env python3
import os
import json
import pickle
import openml
import pickle
import numpy as np
from tqdm import trange
from gama import GamaClassifier
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from gama.postprocessing import EnsemblePostProcessing 
from gama.search_methods import AsynchronousSuccessiveHalving 

if __name__ == "__main__":
  CV_FOL_NAME = "cv_folds"
  openml.config.apikey = "6582f698e8e48968a7566b87fff8a75e"  # set the OpenML Api Key
  benchmark_suite = openml.study.get_suite('OpenML-CC18')  # obtain the benchmark suite

  if not os.path.isdir(CV_FOL_NAME):
    raise BaseException("No CV folds!")

  if not os.path.isdir("open-cc18-gama"):
    os.mkdir("open-cc18-gama")

  for i, t_id in zip(trange(len(benchmark_suite.tasks)), benchmark_suite.tasks):
    if i == 10:
      exit(0)

    try:
      task = openml.tasks.get_task(t_id)
      dataset = task.get_dataset()
      X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
      )

    except Exception as e:
      print("OpenML error", e)

    if len(np.unique(y)) == 2:
      scorer_name = "roc_auc"
      scorer = roc_auc_score
    else:
      scorer_name = "neg_log_loss"
      scorer = log_loss

    with open(f"{CV_FOL_NAME}/{dataset.id}.pkl", "rb") as f:
      cv = pickle.load(f) 
 
    for fold_nr, (train_idx, test_idx) in enumerate(cv):
      try:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        with open("gama_pop.pkl", "rb") as f:
          pop = pickle.load(f)

        clf = GamaClassifier(
          scoring=scorer_name,
          max_total_time=900,
          max_eval_time=240,
          n_jobs=-1,
          #search=AsynchronousSuccessiveHalving(),
          post_processing=EnsemblePostProcessing()
        )

        clf.fit(X_train, y_train, warm_start=pop)
        #clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)
        if scorer_name == "roc_auc":
          score = scorer(y_test, y_proba[:,1], labels=np.unique(y))
        else:
          score = scorer(y_test, y_proba, labels=np.unique(y))

        with open(f"open-cc18-gama/{dataset.id}_{fold_nr}.json", "w") as f:
          fold_res = {
            "data_id": dataset.id,
            "name": dataset.name,
            "metric": scorer_name,
            "score": score
          }
          json.dump(fold_res, f)

      except Exception as e:
        print("Error with fold", e)
        with open(f"open-cc18-gama/{dataset.id}_{fold_nr}.json", "w") as f:
          fold_res = {
            "data_id": dataset.id,
            "name": dataset.name,
            "metric": scorer_name,
            "score": 0 
          }
          json.dump(fold_res, f)
        
      # clean up
      if os.path.isdir("cache/"):
        os.system("rm -rf cache/")

