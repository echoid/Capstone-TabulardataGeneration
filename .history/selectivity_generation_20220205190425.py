import numpy as np
from scipy.spatial import distance
import sys
import math
import pickle
from tqdm import tqdm


#part_id = int(sys.argv[1])


data_file = sys.argv[1]
query_file = data_file

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




#cid = int(sys.argv[1])

# 原始文件，feature文件，label文件，combined文件

data_file = sys.argv[2]
label_file = sys.argv[3]
output_file = sys.argv[4]



data_num = data.shape[0]

# read original data
#data_file = '../../training_feats/glove_50_trainingFeats_part' + str(cid) + '_d64_binary_codes.npy'
#data_file = '../../training_feats/face_d128_2M_trainingFeats_part' + str(cid) + '.txt.npy'


x_dim = data.shape[1]
# current data 维度


#label_file = '../../raw_labels/face_d128_2M_trainingFeats_smallSel_part' + str(cid) + '_rawLabels.npy'
labels = predictions

#print(labels[0], labels[1], len(labels))

selectivity = np.geomspace(0.0001, 1, 40)

tau_max_per_record = len(selectivity)


# save mixlabels file
# current data instance * tau, current dimension + 2

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
