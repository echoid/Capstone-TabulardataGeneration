import numpy as np
from tqdm import tqdm
import pandas as pd
from ctgan.data_transformer import DataTransformer
import json
import sys

origin_path = "dataset/origin/"
config_path = "dataset/configeration/"
save_path = "dataset/transformed/"
dataname = sys.argv[1] + ".csv"

train = pd.read_csv(origin_path + dataname)

with open(config_path + sys.argv[1]+"_config.json", 'r') as f:
  config = json.load(f)
discrete_columns = train.columns[config["one-hot_cols"]].tolist()


transformer = DataTransformer()
transformer.fit(train, discrete_columns)
train = transformer.transform(train)

np.save(save_path+sys.argv[1]+"_transformed.npy",train)

print("Data converted, new dim: ",train.shape)

