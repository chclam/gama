import os
import numpy as np
import fasttext
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.base import ClassifierMixin, BaseEstimator

class FastTextClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self):
    self._estimator_type = "classifier"
    self.classes_ = None
    self.num_labels = None
    self.model_filename = None
    
  def fit(self, X, y):
    data_fn = "cache/test_data.txt"
    data = self.preprocess(X, y)
    self.classes_ = sorted(list(pd.Series(y).unique()))
    pd.set_option('display.max_colwidth', None)
    with open(data_fn, "w+") as out:
      out.write(data.to_string(index=False))
    model = fasttext.train_supervised(data_fn)
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
    # sort the probabilities by label order
    ret = []
    for row_labels, row_probs in zip(labels_pred, probs_pred):
      sorted_by_labels = sorted(list(zip(row_labels, row_probs)), key=lambda x: x[0])
      ret.append([lbl[1] for lbl in sorted_by_labels])
    return np.array(ret)

  def predict_proba(self, X):
    return self.predict(X, ret_proba=True)
  
  def score(self, X, y):
    pass

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

  def preprocess(self, X, y=None):
    X = pd.DataFrame(X).reset_index(drop=True)
    if y is not None:
      y = pd.DataFrame(y).reset_index(drop=True)

    # formatting y to fit fasttext expected format
    ret = X.copy()
    if y is not None:
      ret = pd.concat([ret, y.astype(str).apply(lambda x: "__label__" + x, axis=1)], axis=1)
    # join all columns into one column
    ret = ret.astype(str)
    ret = ret.fillna(" ")
    ret = ret.apply(lambda r: " ".join(v for v in r.values), axis=1)
    ret = ret.str.lower()
    ret = ret.str.replace("<.*?>", " ", regex=True)  # remove html tags
    ret = ret.str.replace("""([\\W])""", " \\1 ", regex=True)  # separate special characters
    ret = ret.str.replace("\\s", " ", regex=True)
    ret = ret.str.replace("[ ]+", " ", regex=True)
    return ret

if __name__ == "__main__":
  # Getting the data set
  dataset = openml.datasets.get_dataset(20)
  X, y, _, _ = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
  )

  #y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
  y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  clf = FastTextClassifier()
  clf.fit(X_train, y_train)
 #print(clf.validate(X_test, y_test))
  print(clf.predict_proba(X_test))
  #import pdb; pdb.set_trace()
