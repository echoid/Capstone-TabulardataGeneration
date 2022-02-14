import numpy as np
from scipy.spatial import distance
from util import sel_generation
import sys
import math
import pickle
from tqdm import tqdm
import pandas as pd
import os
from selnet import *
from data import NumericalField, CategoricalField, Iterator
from data import Dataset

train = pd.read_csv("dataset/train/adult.csv")

config = { 
        "name": "adult_fd_test",
		"train": "adult.csv",
		"gmm_cols":[],
		"normalize_cols":[0,2,4,10,11,12],
		"one-hot_cols":[1,3,5,6,7,8,9,13,14],
		"ordinal_cols":[],
		"model": "VGAN",
		"n_epochs":20,
		"steps_per_epoch":10,
		"n_search": 5,
		"rand_search": "yes",
		"train_method":"VTrain"
	}


n_search = config["n_search"]
search = 1
ratio = 0.9
noise = 0.1
fields = []
col_type = []


for i, col in enumerate(list(train)):

    if i in config["normalize_cols"]:
        fields.append((col,NumericalField("normalize")))
        col_type.append("normalize")
    elif i in config["gmm_cols"]:
        fields.append((col, NumericalField("gmm", n=5)))
        col_type.append("gmm")
    elif i in config["one-hot_cols"]:
        fields.append((col, CategoricalField("one-hot", noise=noise)))
        col_type.append("one-hot")
    elif i in config["ordinal_cols"]:
        fields.append((col, CategoricalField("dict")))
        col_type.append("ordinal")
    else:
        fields.append((col, CategoricalField("binary",noise=noise)))
        col_type.append("binary")


trn = Dataset.split(
    fields = fields,
    path = "dataset/train/",
    train = config["train"],
    format = "csv",
)[0]
trn.learn_convert()
train_it= Iterator.split(
    batch_size = 128,
    train = trn)[0]

train_data = np.array(tf.concat([data for data in train_it], axis=0))

#np.save("dataset/train/adult_converted.npy",train_data)

print("Data converted, new dim: ",train_data.shape)