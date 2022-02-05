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

# data = np.load("dataset/train/adult_converted.npy")
# query = np.load("dataset/train/adult_converted.npy")
# np.random.seed(20)
# sc = np.random.choice(data.shape[0], int(data.shape[0]*0.5), replace=False)
# query = data[sc]
# sel = sel_generation(data,query)
# np.save("dataset/train/adult_converted_sel_50pct.npy",sel)




loss_option = 'huber_log'
partition_option = 'l2'

dataFile = 'dataset/train/adult_converted_sel_full.npy'

valid_file = 'dataset/train/adult_converted_sel_50pct.npy'



train_data = np.load(dataFile)
valid_data = np.load(valid_file)

x_dim = train_data.shape[1]-2
x_reducedim = x_dim
print(x_dim)

tau_part_num = 50

train_original_X = np.array(train_data[:, :x_dim], dtype=np.float32)
train_tau_ = []
for rid in range(train_data.shape[0]):
    t = train_data[rid, x_dim] #hm_to_l2(train_data[rid, x_dim])
    train_tau_.append(t)

train_tau_ = np.array(train_tau_)
train_tau = np.zeros((train_data.shape[0], tau_part_num))
for cid in range(tau_part_num):
    train_tau[:, cid] = train_tau_

train_Y = np.array(train_data[:, -1], dtype=np.float32)



valid_original_X = np.array(valid_data[:, :x_dim], dtype=np.float32)
valid_tau_ = []
for rid in range(valid_data.shape[0]):
    t = valid_data[rid, x_dim] #hm_to_l2(test_data[rid, x_dim])
    valid_tau_.append(t)

valid_tau_ = np.array(valid_tau_)
valid_tau = np.zeros((valid_data.shape[0], tau_part_num))
for cid in range(tau_part_num):
    valid_tau[:, cid] = valid_tau_

valid_Y = np.array(valid_data[:, -1], dtype=np.float32)


unit_len = 100
max_tau = 1 #54.0

hidden_units = [512, 512, 512, 256]
vae_hidden_units = [512, 256, 128]

batch_size = 512
#epochs = 1500
epochs = 120
epochs_vae = 100
learning_rate = 0.00003
log_option = False
tau_embedding_size = 5
original_x_dim = train_original_X.shape[1]
dimreduce_x_dim = x_reducedim

test_data_predictions_labels_file = os.path.join('./test_face_d128_2M_smallSel_huber_log/', 'test_predictions.npy')
valid_data_predictions_labels_file = os.path.join('./test_face_d128_2M_smallSel_huber_log/', 'valid_predictions_labels_one_epoch_')

regression_name = 'adult'
regression_model_dir = 'pretrained_models/sel'


# train
regressor = SelNet(hidden_units, vae_hidden_units, batch_size, epochs, epochs_vae,
                         learning_rate, log_option, tau_embedding_size, original_x_dim, dimreduce_x_dim,
                         test_data_predictions_labels_file, valid_data_predictions_labels_file, regression_name, 
                         regression_model_dir, unit_len, max_tau, tau_part_num, partition_option, loss_option)

regressor.train_vae_dnn(train_original_X, train_tau, train_Y, valid_original_X, valid_tau, valid_Y)
