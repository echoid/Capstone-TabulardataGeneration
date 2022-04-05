from ctgan import CTGANSynthesizer
import pandas as pd
import os
import sys
import json


dataname = sys.argv[1]
path = "dataset/"
data = pd.read_csv(path + "origin/"+ dataname + ".csv")

with open(path +"configeration/" +sys.argv[1]+"_config.json", 'r') as f:
  config = json.load(f)

generated_path = path + "generated/" + dataname + "/selgan/"
try:
        os.makedirs(generated_path)
except:
    pass


discrete_columns = data.columns[config["one-hot_cols"]].tolist()



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

selgan = CTGANSynthesizer(epochs=int(sys.argv[2]),verbose=True,selnet="adult")

selgan.fit(data, discrete_columns,log = generated_path)
selgan.save(generated_path + "selgan_{}_{}.pkl".format(sys.argv[2],sys.argv[3]))

# Synthetic copy
samples = selgan.sample(data.shape[0])

samples.to_csv(generated_path + "selgan_{}_{}.csv".format(sys.argv[2],sys.argv[3]),index=False)