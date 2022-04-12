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



daisy_sel = pd.read_csv(generated_datapath + "sel/sel_1_800_0.csv").sample(n=1000, random_state=1)
daisy_sm = pd.read_csv(generated_datapath + "sel_mean/sel_mean_1_800_0.csv").sample(n=1000, random_state=1)
daisy_KL = pd.read_csv(generated_datapath + "KL/KL_1_800_0.csv").sample(n=1000, random_state=1)
vae = pd.read_csv(generated_datapath + "VAE/sample_data_vae_800_0.csv")
vae.columns = origin_data.columns
vae = vae.sample(n=500, random_state=1)



with open("dataset/configeration/" + sys.argv[1]+"_config.json", 'r') as f:
  config = json.load(f)
discrete_columns = origin_data.columns[config["one-hot_cols"]].tolist()



transformer = DataTransformer()
transformer.fit(origin_data, discrete_columns)
train_data = transformer.transform(origin_data)
test_sel = transformer.transform(daisy_sel)
test_sm = transformer.transform(daisy_sm)
test_kl = transformer.transform(daisy_KL)
test_vae = transformer.transform(vae)






selnet = load_sel(train_data,dataname)




sel_query = query_generation(train_data, test_sel)
#np.save(generated_datapath + "ctgan/ctgan_less_300_sample.npy", ctgan_query)
sm_query = query_generation(train_data, test_sm)
#np.save(generated_datapath + "selgan/selgan_300_less_batch_sample.npy", selgan_query)
kl_query = query_generation(train_data, test_kl)
#np.save(generated_datapath + "ctgan/ctgan_less_300_sample.npy", ctgan_query)
vae_query = query_generation(train_data, test_vae)
#np.save(generated_datapath + "selgan/selgan_300_less_batch_sample.npy", selgan_query)



# train_score = selectivity_evaluation(real_query, selnet)
sel_score = selectivity_evaluation(sel_query, selnet)
sm_score = selectivity_evaluation(sm_query, selnet)
kl_score = selectivity_evaluation(kl_query, selnet)
vae_score = selectivity_evaluation(vae_query, selnet)
# print("Finished")

# print("train score", train_score)

print("generated sel score", sel_score)

print("generated sm score", sm_score)

print("generated kl score", kl_score)

print("generated vae score", vae_score)





