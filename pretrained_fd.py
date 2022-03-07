import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import json


acc_dic = {}
fd_data = pd.read_csv("dataset/train/adult.csv")
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

result = FD_Strong_num.evaluate(Input, Output)
acc_dic["strong_num"] = result
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
result = FD_Weak_num.evaluate(Input, Output)
acc_dic["weak_num"] = result
print(type(result))
FD_Weak_num.save('pretrained_models/weak_num')



print("【Functional Dependiencies 3: Strong Categorical】")
print("【marital-status + sex -> relationship】")

ms = np.array(pd.get_dummies(fd_data["marital-status"])) 
sex = np.array(pd.get_dummies(fd_data["sex"]))
Input = np.concatenate((ms,sex),axis=1)
Output = fd_data["relationship"]
Output_dummy = np.array(pd.get_dummies(fd_data["relationship"]))


def Strong_cate():
  model = keras.Sequential([
    layers.Dense(128, input_dim=9, activation='relu'),
    layers.Dense(8, activation='relu'),
	layers.Dense(6, activation='softmax'),
  ])

  optimizer = keras.optimizers.Adam(learning_rate=0.001)

  model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
               optimizer=optimizer, metrics=['accuracy'])
  return model


FD_Strong_cate = Strong_cate()

FD_Strong_cate.fit(
  Input, Output_dummy,epochs=100,batch_size=256)

loss, acc = FD_Strong_cate.evaluate(Input, Output_dummy)
print(type(acc))
print(acc)
acc_dic["strong_cate"] = np.float64(acc)
FD_Strong_cate.save('pretrained_models/strong_cate')




print("【Functional Dependiencies 4: Weak Categorical】")
print("【education -> occupation】")


Input = np.array(pd.get_dummies(fd_data["education"])) 
Output_dummy = np.array(pd.get_dummies(fd_data["occupation"]))

def Weak_cate():
  model = keras.Sequential([
    layers.Dense(128, input_dim=16, activation='relu'),
    layers.Dense(64, activation='relu'),
	layers.Dense(14, activation='softmax'),
  ])

  optimizer = keras.optimizers.Adam(learning_rate=0.001)

  model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
               optimizer=optimizer, metrics=['accuracy'])
  return model


FD_Weak_cate = Weak_cate()

FD_Weak_cate.fit(
  Input, Output_dummy,epochs=100,batch_size=256)

loss, acc = FD_Weak_cate.evaluate(Input, Output_dummy)
acc_dic["weak_cate"] =  np.float64(acc)

FD_Weak_cate.save('pretrained_models/weak_cate')


print("All Pretrained FD models Saved!")


 
with open("pretrained_models/base_acc.json", "w") as outfile:
    json.dump(acc_dic, outfile)