from ctgan import CTGANSynthesizer
from ctgan import load_demo
import pandas as pd
import os
import sys
import json


dataname = sys.argv[1]
path = "dataset/"
data = pd.read_csv(path + "origin/"+ dataname + ".csv")

with open(path +"configeration/" +sys.argv[1]+"_config.json", 'r') as f:
  config = json.load(f)

generated_path = path + "generated/" + dataname + "/ctgan/"
try:
        os.makedirs(generated_path)
except:
    pass


discrete_columns = data.columns[config["one-hot_cols"] + config["ordinal_cols"]].tolist()



# data = load_demo()

# # Names of the columns that are discrete
# discrete_columns = [
#     'workclass',
#     'education',
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'sex',
#     'native-country',
#     'income'
# ]

ctgan = CTGANSynthesizer(epochs=int(sys.argv[2]))

ctgan.fit(data, discrete_columns)

# Synthetic copy
samples = ctgan.sample(data.shape[0])

samples.to_csv(generated_path + "ctgan_" + sys.argv[2] + ".csv",index=False)