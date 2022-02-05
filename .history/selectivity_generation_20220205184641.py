import numpy as np
from scipy.spatial import distance
import sys
import math
import pickle
from tqdm import tqdm

#part_id = int(sys.argv[1])


data_file = sys.argv[1]
query_file = sys.argv[2]
result_file = sys.argv[3]


print(data_file)

#data_file = '../real_data/face_d128_2M_originalData.npy'
data = np.load(data_file)

data_num = data.shape[0]
print("Successful loaded!")

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

#result_file = '../raw_labels/face_d128_2M_trainingFeats_smallSel_part' + str(part_id) + '_rawLabels.npy'
np.save(result_file, predictions)