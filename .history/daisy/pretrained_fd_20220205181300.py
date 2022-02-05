import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pandas as pd


def train_and_save(Input,Output,model_func):
    model = model_func()

fd_data = pd.read_csv("./dataset/train/adult_train.csv")
Input = fd_data[["education-num","capital-gain"]]
Output = fd_data["hours-per-week"]

print("【Functional Dependiencies 1: Strong Numerical】")
print("【Education-num + capital-gain -> hours-per-week】")

def Strong_num():
  model = keras.Sequential([
    layers.Dense(64, input_dim=2, activation='relu'),
    layers.Dense(8, activation='relu'),
	  layers.Dense(1),
  ])

  optimizer = keras.optimizers.Adam(learning_rate=0.001)

  model.compile(loss="mae",
               optimizer=optimizer)
  return model

FD_Strong_num = Strong_num()

FD_Strong_num.fit(Input, Output,epochs=100,batch_size=256)

FD_Strong_num.save('pretrained_models/strong_num')
print("Strong-num pretrained model saved")
# if want load model
# FD_Strong_num = tf.keras.models.load_model('pretrained_models/strong_num')

# result = FD_Strong_num.predict(Input)


print("【Functional Dependiencies 2: Weak Numerical】")
print("【Education-num -> age】")


Input = fd_data["education-num"]
Output = fd_data["age"]

def Weak_num():
  model = keras.Sequential([
    layers.Dense(64, input_dim=1, activation='relu'),
    layers.Dense(8, activation='relu'),
	  layers.Dense(1),
  ])

  optimizer = keras.optimizers.Adam(learning_rate=0.001)

  model.compile(loss="mae",
               optimizer=optimizer)
  return model


FD_Weak_num = Weak_num()
FD_Weak_num.fit(Input, Output,epochs=100,batch_size=256)
FD_Weak_num.save('pretrained_models/weak_num')
print("Weak-num pretrained model saved")

