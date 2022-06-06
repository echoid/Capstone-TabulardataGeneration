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



ctgan = pd.read_csv(generated_datapath + "ctgan/ctgan_less_300.csv").sample(n=1000, random_state=1)
selgan = pd.read_csv(generated_datapath + "selgan/selgan_300_full_batch.csv").sample(n=1000, random_state=1)



with open("dataset/configeration/" + sys.argv[1]+"_config.json", 'r') as f:
  config = json.load(f)
discrete_columns = origin_data.columns[config["one-hot_cols"]].tolist()



transformer = DataTransformer()
transformer.fit(origin_data, discrete_columns)
train_data = transformer.transform(origin_data)
test_ctgan = transformer.transform(ctgan)
test_selgan = transformer.transform(selgan)





selnet = load_sel(train_data,dataname)




ctgan_query = query_generation(train_data, test_ctgan)
#np.save(generated_datapath + "ctgan/ctgan_less_300_sample.npy", ctgan_query)
selgan_query = query_generation(train_data, test_selgan)
#np.save(generated_datapath + "selgan/selgan_300_less_batch_sample.npy", selgan_query)



print(ctgan_query.shape)
print(selgan_query.shape)

# train_score = selectivity_evaluation(real_query, selnet)
ctgan_score = selectivity_evaluation(ctgan_query, selnet)
selgan_score = selectivity_evaluation(selgan_query, selnet)

# print("Finished")

# print("train score", train_score)

print("generated ctgan score", ctgan_score)

print("generated selnet score", selgan_score)






# ctgan_score = selectivity_evaluation(generated_query_ctgan, selnet)
# selgan_score = selectivity_evaluation(generated_query_selgan, selnet)

# print("Finished")
# #print("train score", train_score)

# print("1 generated ctgan score", ctgan_score)

# print("1 generated selnet score", selgan_score)
# # ## model loaded


# ctgan_score = selectivity_evaluation(load_query_ctgan, selnet)
# selgan_score = selectivity_evaluation(load_query_selgan, selnet)

# print("Finished")
# #print("train score", train_score)

# print("loaded ctgan score", ctgan_score)

# print("loaded selnet score", selgan_score)
# # ## model loaded

# load_query_ctgan = np.load(generated_datapath + "ctgan/ctgan_less_300_sample.npy", allow_pickle=True)
# load_query_selgan = np.load(generated_datapath + "selgan/selgan_300_less_batch_sample.npy", allow_pickle=True)

# ctgan_score = selectivity_evaluation(load_query_ctgan, selnet)
# selgan_score = selectivity_evaluation(load_query_selgan, selnet)

# print("Finished")
# #print("train score", train_score)

# print("2 loaded ctgan score", ctgan_score)

# print("2 loaded selnet score", selgan_score)


# # ## load pre-trained model

# ctgan_score = selectivity_evaluation(generated_query_ctgan, selnet)
# selgan_score = selectivity_evaluation(generated_query_selgan, selnet)

# print("Finished")
# #print("train score", train_score)

# print("2 generated ctgan score", ctgan_score)

# print("2 generated selnet score", selgan_score)



# train_data_sample = train_data.sample(n=1000, random_state=1)

# generated_query_real = query_generation(train_data, train_data_sample)
# np.save(generated_datapath + "query_sample.npy", generated_query_real)
# load_query_real = np.load(generated_datapath + "query_sample.npy", allow_pickle=True)







# generated_query_ctgan = query_generation(train_data, test_ctgan)
# np.save(generated_datapath + "ctgan/ctgan_less_300_sample.npy", generated_query_ctgan)
# load_query_ctgan = np.load(generated_datapath + "ctgan/ctgan_less_300_sample.npy", allow_pickle=True)


# generated_query_selgan = query_generation(train_data, test_selgan)
# np.save(generated_datapath + "selgan/selgan_300_less_batch_sample.npy", generated_query_selgan)
# load_query_selgan = np.load(generated_datapath + "selgan/selgan_300_less_batch_sample.npy", allow_pickle=True)



# if (load_query_selgan == generated_query_selgan).all():
#   print("They are the same_selgan")
# else:
#   print("they are not same_selgan")
#   print(load_query_selgan.shape,generated_query_selgan.shape)


# if (load_query_ctgan == generated_query_ctgan).all():
#   print("They are the same_ctgan")
# else:
#   print("they are not same_ctgan")
#   print(load_query_ctgan.shape,generated_query_ctgan.shape)
