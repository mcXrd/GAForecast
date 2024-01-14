from ..models.gaforecast import GAForecastBinaryClassifier
import pandas as pd
import os
from sklearn.metrics import accuracy_score


def _get_data_directory():
    test_directory = os.path.abspath(os.path.dirname(__file__))
    data_directory = os.path.join(test_directory, "..", "data")
    return data_directory

def _prepare_data(train_or_test_word, frac=0.02):
  data_file = os.path.join(_get_data_directory(), f"stock_details_{train_or_test_word}.parquet")
  df = pd.read_parquet(data_file)
  df = df.sample(frac=frac)
  del df["target"]
  labels = df["binary_target"]
  del df["binary_target"]
  data = df
  return data, labels

def test_fit_and_predict():
    train_data, train_labels = _prepare_data("train")
    test_data, test_labels = _prepare_data("test")
    model = GAForecastBinaryClassifier(epochs=2)
    model.fit(train_data, train_labels)

    predictions = model.predict(test_data)
    accuracy_score(test_labels, predictions)




