import os
import numpy as np
import fasttext
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.base import ClassifierMixin, BaseEstimator
from time import time

class FastTextClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, lr=0.1, epoch=5, wordNgrams=1, minn=0, maxn=0, pretrainedVectors=""):
    self._estimator_type = "classifier"
    self.classes_ = None
    self.model_filename = None
    self.lr = lr
    self.epoch = epoch
    self.wordNgrams = wordNgrams
    self.minn = minn
    self.maxn = maxn
    self.pretrainedVectors=pretrainedVectors
    
  def fit(self, X, y, classes=None):
    '''
    TODO: Label encode y
    '''
    if not os.path.isdir("cache"):
      os.mkdir("cache")
    data_fn = f"cache/test_data{time()}.txt"

    if self.classes_ is None:
      self.classes_ = sorted(np.unique(y))
    pd.set_option('display.max_colwidth', None) # do this so that .to_string() actually converts all data to string
    data = self.preprocess(X, y, data_fn=data_fn)
#    with open(data_fn, "w+") as out:
#      out.write(data.to_string(index=False, header=False))
    model = fasttext.train_supervised(data_fn, lr=self.lr, epoch=self.epoch, wordNgrams=self.wordNgrams, minn=self.minn, maxn=self.maxn, pretrainedVectors=self.pretrainedVectors)
    self.model_filename = f"cache/ft_model_{time()}.bin"
    # save and load the model due to issues with multiprocessing when passing on the fit model.
    model.save_model(self.model_filename)
    return self

  def predict(self, X, ret_proba=False):
    if self.model_filename is None:
      raise Exception("Model is not fitted yet. Please fit the model first.")
    pd.set_option('display.max_colwidth', None)
    data = self.preprocess(X)
    data = data.split("\n") # split the data into an array of rows
    model = fasttext.load_model(self.model_filename)
    classes_pred, probs_pred = model.predict(data, k=len(self.classes_) if ret_proba else 1)
    if not ret_proba:
      # flatten list and get rid of "__label__" prefix
      # TODO: conversion to int is only needed if label encoder is used.
      return np.array([int(x[0].replace("__label__", "")) for x in classes_pred])
    else:
      ret = []
      for row_classes, row_probs in zip(classes_pred, probs_pred):
        sorted_by_classes = sorted(list(zip(row_classes, row_probs)), key=lambda x: x[0])
        probas = [lbl[1] for lbl in sorted_by_classes]
        # make sure probabilities nicely sum up to 1 by subtracting the excess from the highest class probability.
        # highest because subtracting from lowest can result in negative numbers.
        probas[np.argmax(probas)] -= sum(probas) - 1.
        ret.append(probas)
      return np.array(ret)

  def predict_proba(self, X):
    return self.predict(X, ret_proba=True)
  
  def score(self, X, y_true):
    y_pred = self.predict(X)
    return accuracy_score(y_true, y_pred)

  def preprocess(self, X, y=None, del_spec_chars=True, data_fn=f"cache/test_data{time()}.txt") -> str:
    X = pd.DataFrame(X, columns=X.columns if isinstance(X, pd.DataFrame) else None).reset_index(drop=True)
    data = X.copy()
    data = data.astype(str)
    data = data.fillna(" ")
    if y is not None:
      # formatting y to fit fasttext expected format
      y = pd.DataFrame(y, columns=[0]).reset_index(drop=True)
      y.name = None
      y[1] = y[0]
      y[0] = "__label__"
      y = y.astype(str).apply(lambda r: "".join(v for v in r.values), axis=1)
      data = pd.concat([data, y], axis=1)
      # Sloppy hack for appending "__label__" faster to all values  
    '''
    TODO: write a bash script to make this faster
    '''
    pd.set_option('display.max_colwidth', None) # do this so that .to_string() actually converts all data to string
    data.to_csv(data_fn, header=None, index=None, sep=' ', mode='w')
    # call bash script for preprocessing
    os.system(f"gama/configuration/preprocess_X.sh {data_fn}")
    with open(data_fn, "r") as f:
      ret = f.read()
    if ret[-1] == "\n" and len(ret) > 0:
      ret = ret[:(len(ret) - 1)] # remove trailing newline
    return ret 

