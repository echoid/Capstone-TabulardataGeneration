import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pandas as pd

fd_data = pd.read_csv("adult_train.csv")
Input = fd_data[["education-num","capital-gain"]]
Output = fd_data["hours-per-week"]