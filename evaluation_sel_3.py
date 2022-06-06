import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from eval_util import make_compare_plot,plot_cdf,plot_pdf,make_prediction_diff,make_prediction,DCR, hitting_rate, make_clustering,convert_type
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import numpy
from tqdm import tqdm
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import os
import glob
import sys
import json
from selgan.util_eval import load_sel, selectivity_evaluation, query_generation
from data_transformer import DataTransformer

#matplotlib inline

dataname = sys.argv[1]







generated_datapath = "dataset/generated/{}/".format(dataname)

origin_data = pd.read_csv("dataset/origin/{}.csv".format(dataname))

if (dataname =="adult"):
  origin_data = origin_data.drop(columns=['income'])



with open("dataset/configeration/" + sys.argv[1]+"_config.json", 'r') as f:
  config = json.load(f)
discrete_columns = origin_data.columns[config["one-hot_cols"]].tolist()



transformer = DataTransformer()
transformer.fit(origin_data, discrete_columns)
train_data = transformer.transform(origin_data)
selnet = load_sel(train_data,dataname)



if (dataname =="adult") or  (dataname =="credit"):
    tablegan = pd.read_csv(generated_datapath + "tablegan/tablegan_{}_fake.csv".format(dataname)).sample(n=1000, random_state=1)
    test_tablegan = transformer.transform(tablegan)
    tablegan_query = query_generation(train_data, test_tablegan)
    table_score = selectivity_evaluation(tablegan_query, selnet)
    print("generated table score", table_score)

# if not dataname == "ticket":
#     octgan = pd.read_csv(generated_datapath + "octgan/octgan_{}_less.csv".format(dataname)).sample(n=1000, random_state=1)
#     test_octgan = transformer.transform(octgan)
#     octgan_query = query_generation(train_data, test_octgan)
#     oct_score = selectivity_evaluation(octgan_query, selnet)
#     print("generated oct score", oct_score)


# print("Finished")

# print("train score", train_score)










