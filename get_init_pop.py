#!/usr/bin/env python3
from gama import GamaClassifier
from sklearn.base import is_classifier
from gama.configuration.fasttextclassifier import FastTextClassifier
import pickle


if __name__ == "__main__":
  with open("gama_pop_with_ft.pkl", "rb") as f:
    pop = pickle.load(f) 

  for prim in pop:
    for step in prim.pipeline:
      if is_classifier(step) and isinstance(step, FastTextClassifier):
        print(step)

  exit(0)

  while True:
    clf = GamaClassifier()
    init_pop = clf._get_init_pop(55)
    ft_pop = [prim for prim in init_pop for step in prim.pipeline if is_classifier(step) and isinstance(step, FastTextClassifier)]
    init_pop = [prim for prim in init_pop for step in prim.pipeline if is_classifier(step) and not isinstance(step, FastTextClassifier)]

    if len(init_pop) == 50:
      break

  with open("gama_pop.pkl", "wb") as f:
    pickle.dump(init_pop, f) 

  with open("gama_pop_with_ft.pkl", "wb") as f:
    pickle.dump(init_pop + ft_pop, f) 

