#!/usr/bin/env python3
import os
import pickle
import openml
from tqdm import trange
from sklearn.model_selection import StratifiedKFold 

if __name__ == "__main__":
#  FOL_NAME = "cv_folds"
#  openml.config.apikey = "6582f698e8e48968a7566b87fff8a75e"  # set the OpenML Api Key
#  benchmark_suite = openml.study.get_suite('OpenML-CC18')  # obtain the benchmark suite
#
#  if not os.path.isdir(FOL_NAME):
#    os.mkdir(FOL_NAME)
#
#  for _, t_id in zip(trange(len(benchmark_suite.tasks)), benchmark_suite.tasks):
#    task = openml.tasks.get_task(t_id)
#    dataset = task.get_dataset()
#    X, y, _, _ = dataset.get_data(
#      target=dataset.default_target_attribute, dataset_format="dataframe"
#    )
#    cv = list(StratifiedKFold(n_splits=5, random_state=None, shuffle=True).split(X, y))
#    with open(f"{FOL_NAME}/{dataset.id}.pkl", "wb") as f:
#      pickle.dump(cv, f)

  FOL_NAME = "cv_folds"
  datasets = {
#    "openpayments": 42738,
#    "midwest": 42530,
#    "traffic_violations": 42132,
#    "road_safety": 42803,
#    "beerreviews": 42078,
#    "drug_directory": 43044,
#    "kickstarter": 42076,
    #"cacao_flavors": 42133,
    "met_objects": 'MET_OBJECTS'
  }

  
  for dataset_name, d_id in datasets.items():
    fn = f"{FOL_NAME}/{d_id}.pkl"
    if not os.path.isfile(fn):
      with open(f"prepped_data/{d_id}_X.pkl", "rb") as f:
        X = pickle.load(f)
      with open(f"prepped_data/{d_id}_y.pkl", "rb") as f:
        y = pickle.load(f)
      try:
        cv = list(StratifiedKFold(n_splits=5, random_state=None, shuffle=True).split(X, y))
      except Exception:
        import pdb; pdb.set_trace()
      fn = f"{FOL_NAME}/{d_id}.pkl"
      with open(fn, "wb") as f:
        pickle.dump(cv, f)

