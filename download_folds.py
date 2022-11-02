#!/usr/bin/env python3
import os
import pickle
import openml
from tqdm import trange
from sklearn.model_selection import StratifiedKFold 

if __name__ == "__main__":
  FOL_NAME = "cv_folds"
  openml.config.apikey = "6582f698e8e48968a7566b87fff8a75e"  # set the OpenML Api Key
  benchmark_suite = openml.study.get_suite('OpenML-CC18')  # obtain the benchmark suite

  if not os.path.isdir(FOL_NAME):
    os.mkdir(FOL_NAME)

  for _, t_id in zip(trange(len(benchmark_suite.tasks)), benchmark_suite.tasks):
    task = openml.tasks.get_task(t_id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(
      target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    cv = list(StratifiedKFold(n_splits=5, random_state=None, shuffle=True).split(X, y))
    with open(f"{FOL_NAME}/{dataset.id}.pkl", "wb") as f:
      pickle.dump(cv, f)

