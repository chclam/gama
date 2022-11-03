#!/usr/bin/env python3
from gama import GamaClassifier
import pickle


clf = GamaClassifier()
init_pop = clf._get_init_pop()
for prim in init_pop:
  print(str(prim))

with open("pop.pkl", "wb") as f:
  pickle.dump(init_pop, f) 

