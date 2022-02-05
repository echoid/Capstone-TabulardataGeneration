import numpy as np
from scipy.spatial import distance
import sys
import math
import pickle
from tqdm import tqdm
import pandas as pd


def convert_type(data,columns):
    data[columns] = data[columns].astype('category')
    new_data = data.copy()
    for col in columns:
        new_data[col] = data[col].cat.codes
    return new_data


def convert_back(data,converted,columns):
    data[columns] = data[columns].astype('category')
    converted[columns] = converted[columns].astype('category')
    for col in columns:
        converted[col] = data[col].cat.categories[converted[col].cat.codes]
    return converted


def sel_generation(data, query):
    data_file = sys.argv[1]
    query_file = sys.argv[2]
    output_file = sys.argv[3]

    data = np.load(data_file)
queries = np.load(query_file)

data_num = data.shape[0]

# for train
#query_file = '../training_feats/face_d128_2M_trainingFeats_part' + str(part_id) + '.txt.npy'
queries = np.load(query_file)
predictions = []
selectivity = np.geomspace(0.0001, 1, 40)
# generate 40 nodes

# total number of instances(queries)
for rid in tqdm(range(queries.shape[0])):

predict = []
predict_1 = []
_query = queries[rid: rid + 1]

# the distance between each query and database 
# 每一条query都有30000个result，记录和原始数据的distance
res = distance.cdist(data, _query, 'cosine')
res = sorted(np.hstack(res))

# generate training data according to selectivity
for sel in selectivity:
    
    # _label = int(data_num * sel / 1) should be 1 or 100?
    _label = int(data_num * sel / 100)
    _label_1 = int(data_num * sel / 1)

    
    #assert (_label - 1) >= 0, "Labels should be >= 1"
    if (_label - 1) < 0:
        _label = 1
    predict.append(res[_label - 1])
    predict_1.append(res[_label_1 - 1])
    
predictions.append(predict)

predictions = np.array(predictions)




data_num = data.shape[0]
x_dim = data.shape[1]
labels = predictions


selectivity = np.geomspace(0.0001, 1, 40)

tau_max_per_record = len(selectivity)




data_mixlabels = np.zeros((data.shape[0] * tau_max_per_record, data.shape[1] + 1 + 1))


sc_f = []

selected_tau = 3

for rid in tqdm(range(data.shape[0])):
# loop each data 
r_ = data[rid]
# instance
np.random.seed(rid)

# 从40里选3个tau
sc_ = np.random.choice(tau_max_per_record, selected_tau, replace=False)
# 找到最大的tau
t_max_script = np.max(sc_)


for i in range(t_max_script):
    
    tau = labels[rid, i]
    # tau = 第rid条数据的第i个tau value
    _label = int(data_num * selectivity[i] / 100)
    if (_label - 1) < 0:
        _label = 1
    
    
    data_mixlabels[rid * tau_max_per_record + i, :x_dim] = data[rid]
    
    data_mixlabels[rid * tau_max_per_record + i, x_dim] = tau
    data_mixlabels[rid * tau_max_per_record + i, x_dim + 1] = _label
    sc_f.append(rid * tau_max_per_record + i)
    #print("row:",rid * tau_max_per_record + i,"instance:",rid,"tau:",i,"tau value:",tau,"label:",_label)


data_mixlabels = np.array(data_mixlabels, dtype=np.float32)
data_mixlabels = data_mixlabels[sc_f]

data_mixlabels = np.unique(data_mixlabels, axis=0)


np.save(output_file, data_mixlabels)
