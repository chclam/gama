import numpy as np
import fasttext
import pandas as pd
from sklearn.metrics import accuracy_score 
from datetime import datetime
from sklearn.base import ClassifierMixin, BaseEstimator

class FastTextClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, lr=0.1, epoch=5, wordNgrams=1):
    self._estimator_type = "classifier"
    self.classes_ = None
    self.model_filename = None
    self.lr = lr
    self.epoch = epoch
    self.wordNgrams = wordNgrams
    
  def fit(self, X, y):
    data_fn = "cache/test_data.txt"
    data = self.preprocess(X, y)
    self.classes_ = sorted(list(pd.Series(y).unique()))
    pd.set_option('display.max_colwidth', None) # do this so that .to_string() actually converts all data to string
    with open(data_fn, "w+") as out:
      out.write(data.to_string(index=False))
    model = fasttext.train_supervised(data_fn, lr=self.lr, epoch=self.epoch, wordNgrams=self.wordNgrams)
    self.model_filename = f"cache/ft_model_{datetime.now()}.bin"
    model.save_model(self.model_filename)

  def predict(self, X, ret_proba=False):
    if self.model_filename is None:
      raise Exception("Model is not fitted yet. Please fit the model first.")
    pd.set_option('display.max_colwidth', None)
    data = self.preprocess(X).to_string(index=False)
    data = data.split("\n") # split the data into an array of rows
    model = fasttext.load_model(self.model_filename)
    labels_pred, probs_pred = model.predict(data, k=len(self.classes_) if ret_proba else 1)
    if not ret_proba:
      # flatten list and get rid of "__label__" prefix
      # TODO: conversion to int is only needed if label encoder is used.
      return [int(x[0].replace("__label__", "")) for x in labels_pred] 
    else:
      ret = []
      for row_labels, row_probs in zip(labels_pred, probs_pred):
        sorted_by_labels = sorted(list(zip(row_labels, row_probs)), key=lambda x: x[0])
        ret.append([lbl[1] for lbl in sorted_by_labels])
      return np.array(ret)

  def predict_proba(self, X):
    return self.predict(X, ret_proba=True)
  
  def score(self, X, y_true):
    y_pred = self.predict(X)
    return accuracy_score(y_true, y_pred)

  def validate(self, X, y):
    # refactor
    if self.model_filename is None:
      raise Exception("Model is not trained. Please train the model using 'fit' before validating the model.")
    data = self.preprocess(X, y)
    val_data_name = f"cache/validation_set_{datetime.now()}"
    with open(val_data_name, 'w') as out:
      out.write(data.to_string(index=False))
    model = fasttext.load_model(self.model_filename)
    # get rid of this
    print(model.test(val_data_name, k=5))

  def preprocess(self, X, y=None, del_spec_chars=False):
    X = pd.DataFrame(X, columns=X.columns if isinstance(X, pd.DataFrame) else None).reset_index(drop=True)
    if y is not None:
      y = pd.DataFrame(y).reset_index(drop=True)
    # formatting y to fit fasttext expected format
    ret = X.copy()
    # add column name in front of each value
    ret = ret.astype(str)
    ret = ret.fillna(" ")
    if y is not None:
      # Sloppy hack for appending "__label__" faster to all values  
      y[1] = y[0]
      y[0] = "__label__"
      y = y.astype(str).apply(lambda r: "".join(v for v in r.values), axis=1)
      ret = pd.concat([ret, y], axis=1)
    # join all columns into one column
    ret = ret.apply(lambda r: " ".join(v for v in r.values), axis=1)
    ret = ret.str.lower()
    if del_spec_chars:
      # Don't do this by default because regex is slow, esp. on big data sets
      ret = ret.str.replace("<.*?>", " ", regex=True)  # remove html tags
      ret = ret.str.replace("""([\\W])""", " \\1 ", regex=True)  # separate special characters
      ret = ret.str.replace("\\s", " ", regex=True)
      ret = ret.str.replace("[ ]+", " ", regex=True)
    return ret
