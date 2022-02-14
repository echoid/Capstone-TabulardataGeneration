import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from data import NumericalField, CategoricalField, Iterator,Dataset
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from selnet import *
import json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import os 




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


def sel_generation(data, queries):

    data_num = data.shape[0]
    x_dim = queries.shape[1]
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


    labels = predictions

    tau_max_per_record = len(selectivity)

    data_mixlabels = np.zeros((queries.shape[0] * tau_max_per_record, queries.shape[1] + 1 + 1))


    sc_f = []

    selected_tau = 3

    for rid in tqdm(range(queries.shape[0])):
        # loop each data 
        r_ = queries[rid]
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
            
            
            data_mixlabels[rid * tau_max_per_record + i, :x_dim] = queries[rid]
            
            data_mixlabels[rid * tau_max_per_record + i, x_dim] = tau
            data_mixlabels[rid * tau_max_per_record + i, x_dim + 1] = _label
            sc_f.append(rid * tau_max_per_record + i)
            #print("row:",rid * tau_max_per_record + i,"instance:",rid,"tau:",i,"tau value:",tau,"label:",_label)
        

    data_mixlabels = np.array(data_mixlabels, dtype=np.float32)
    data_mixlabels = data_mixlabels[sc_f]

    data_mixlabels = np.unique(data_mixlabels, axis=0)


    return data_mixlabels





def to_df(data,dataset):
    samples = data.reshape(data.shape[0], -1)
    samples = samples[:,:dataset.dim]
    samples = samples.cpu()
    sample_table = dataset.reverse(samples.detach().numpy())
    df = pd.DataFrame(sample_table,columns=dataset.columns)
    return df








def compute_kl(real, pred):
    return torch.sum((torch.log(pred + 1e-4) - torch.log(real + 1e-4)) * pred)

def KL_Loss(x_fake, x_real, col_type, col_dim):
    kl = 0.0
    sta = 0
    end = 0
    for i in range(len(col_type)):
        dim = col_dim[i]
        sta = end
        end = sta+dim
        fakex = x_fake[:,sta:end]
        realx = x_real[:,sta:end]
        if col_type[i] == "gmm":
            fake2 = fakex[:,1:]
            real2 = realx[:,1:]
            # column sum
            dist = torch.sum(fake2, dim=0)

            dist = dist / torch.sum(dist)
            real = torch.sum(real2, dim=0)
            real = real / torch.sum(real)
            kl += compute_kl(real, dist)
        else:
            dist = torch.sum(fakex, dim=0)
            dist = dist / torch.sum(dist)
            
            real = torch.sum(realx, dim=0)
            real = real / torch.sum(real)
            
            kl += compute_kl(real, dist)
    return kl


def mean_Loss(x_fake, x_real, col_type, col_dim):
    mean = 0.0
    sta = 0
    end = 0
    for i in range(len(col_type)):
        dim = col_dim[i]
        sta = end
        end = sta+dim
        fakex = x_fake[:,sta:end]
        realx = x_real[:,sta:end]
        if col_type[i] == "gmm":
            fake2 = fakex[:,1:]
            real2 = realx[:,1:]
            dist = torch.mean(fake2, dim=0)
            dist = dist / torch.sum(dist)
            real = torch.mean(real2, dim=0)
            real = real / torch.sum(real)
            mean += torch.sum(abs(real - dist))
        else:
            dist = torch.mean(fakex, dim=0)
            dist = dist / torch.sum(dist)
            
            real = torch.mean(realx, dim=0)
            real = real / torch.sum(real)
            mean += torch.sum(abs(real - dist))
    return mean


def plot_pdf(data,label,bin=10):
    count, bins_count = np.histogram(data, bins=10)
    pdf = count / sum(count)

    plt.plot(bins_count[1:], pdf, label=label)

def plot_cdf(data,label,bin=10):
    count, bins_count = np.histogram(data, bins=10)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, label=label)

def make_compare_plot(datasets,col_name, function, names, title):
    for i in range(len(datasets)):
        data = datasets[i]
        function(data[col_name],names[i])
    plt.title(title)
    plt.legend()
    plt.show()


def convert_type(data,columns):
    data = data[columns].astype('category')
    for col in columns:
        data[col] = data[col].cat.codes
    return data


def make_prediction(response, response_type,training_data, test,dataset):

    result = []

    train_data_y = training_data[response].astype("float64")
    train_data_X = training_data.drop(columns=[response,"label","fnlwgt"]).astype("float64")

    test_data_X = test.drop(columns=[response,"label","fnlwgt"]).astype("float64")
    test_data_y = test[response].astype("float64")


    if response_type == "clf":

        clf = xgb.XGBClassifier(eval_metric='mlogloss')

    else:

        clf = xgb.XGBRegressor(eval_metric='mlogloss')

    clf.fit(train_data_X, train_data_y)
    result.append(clf.score(test_data_X,test_data_y))


    for data in dataset:

        train_X = data.drop(columns=[response,"label","fnlwgt"]).astype("float64")
        train_y = data[response].astype("float64")
        
        try:
            clf.fit(train_X, train_y)
            result.append(clf.score(test_data_X,test_data_y))
        except:
            result.append(np.nan)

    return result





def make_prediction_diff(response, response_type,training_data, test,dataset):

    result = []

    train_data_y = training_data[response].astype("float64")
    train_data_X = training_data.drop(columns=[response,"label","fnlwgt"]).astype("float64")

    test_data_X = test.drop(columns=[response,"label","fnlwgt"]).astype("float64")
    test_data_y = test[response].astype("float64")


    if response_type == "clf":

        clf = xgb.XGBClassifier(eval_metric='mlogloss')

    else:

        clf = xgb.XGBRegressor(eval_metric='mlogloss')

    clf.fit(train_data_X, train_data_y)
    ground_truth = clf.score(test_data_X,test_data_y)
    result.append(abs(ground_truth - ground_truth))


    for data in dataset:

        train_X = data.drop(columns=[response,"label","fnlwgt"]).astype("float64")
        train_y = data[response].astype("float64")
        
        try:
            clf.fit(train_X, train_y)
            result.append(abs(ground_truth - clf.score(test_data_X,test_data_y)))
        except:
            result.append(np.nan)

    return result




def make_clustering(training_data,test,dataset,n = 5):
    NMI = []

    train_data_x = training_data.drop(columns=["label","fnlwgt","income"]).astype("float64")

    test_data_X = test.drop(columns=["label","fnlwgt","income"]).astype("float64")
    
    kmeans = KMeans(n_clusters=n, random_state=0).fit(train_data_x)

    ground_truth = kmeans.predict(test_data_X)

    NMI.append(normalized_mutual_info_score(ground_truth, ground_truth))

    for data in dataset:
        data = data.drop(columns=["label","fnlwgt","income"]).astype("float64")
        kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
        result = kmeans.predict(test_data_X)
        NMI.append(normalized_mutual_info_score(ground_truth, result))




    return NMI





def hitting_rate(dataset):
 
    # Data_sample
    train_data_sample_X = dataset[0].drop(columns=["label","fnlwgt","income"]).astype("float64")


    distance = []
    for dataframe in dataset[1:]:
        dataframe = dataframe.drop(columns=["label","capital-gain","fnlwgt","income"]).astype("float64")
        sampled = dataframe.sample(n=500, random_state=1)
        sim_item = 0
        for row_sample in tqdm(sampled.iterrows()):
            for row_full in train_data_sample_X.iterrows():
                row_sample_data = row_sample[1]
                row_full_data = row_full[1]
                if ((row_sample_data["age"] == row_full_data["age"]) and
                (row_sample_data["workclass"] == row_full_data["workclass"]) and 
                (row_sample_data["education"] == row_full_data["education"]) and
                (row_sample_data["marital-status"] == row_full_data["marital-status"]) and
                (row_sample_data["occupation"] == row_full_data["occupation"]) and
                (row_sample_data["relationship"] == row_full_data["relationship"]) and
                (row_sample_data["race"] == row_full_data["race"]) and
                (row_sample_data["sex"] == row_full_data["sex"]) and
                (row_sample_data["native-country"] == row_full_data["native-country"]) 
                #and(row_sample_data["income"] == row_full_data["income"])
                ) :
                    sim_item +=1
                    break
        distance.append(sim_item)
    return distance 




def DCR(dataset):

    train_data_sample_X = dataset[0].drop(columns=["label","fnlwgt","income"]).astype("float64")

    scaler = MinMaxScaler()
    scaler.fit(train_data_sample_X)
    train_data_sample_X = pd.DataFrame(scaler.transform(train_data_sample_X))

    mini = []
    maxi = []

    for data in dataset[1:]:
        distance = []

        data = data.drop(columns=["label","fnlwgt","income"]).astype("float64")

        data = pd.DataFrame(scaler.transform(data))
        sampled = data.sample(n=100, random_state=1)
        
        for row_sample in sampled.iterrows():
            for row_full in train_data_sample_X.iterrows():
                row_sample_data = row_sample[1]
                row_full_data = row_full[1]
                distance.append(np.linalg.norm(row_sample_data-row_full_data))
        mini.append(min(distance))
        maxi.append(max(distance))
    return (maxi,mini) 


def fd_calculated(df_fake,fd_model,y_fake):

    path = "pretrained_models/"
    model = tf.keras.models.load_model(path + fd_model)

    with open(path+"base_acc.json", 'r') as f:
        acc_dict = json.load(f)

    y_fake = y_fake.cpu().detach().numpy()
    
    if fd_model == "strong_num":
        Input = df_fake[["education-num","capital-gain"]].astype('float32')
        Output = df_fake["hours-per-week"]


        fd = model.evaluate(Input,y_fake,verbose=False) - acc_dict[fd_model]

        # predicted = model.predict(Input).flatten()

        # fd  = sum(abs(predicted-Output))/y_fake.shape[0]


        return fd

    if fd_model == "weak_num":
        Input = df_fake["education-num"].astype('float32')
        #Output = df_fake["age"]

        # predicted = model.predict(Input).flatten()
        # fd  = sum(abs(predicted-Output))/y_fake.shape[0]

        fd = model.evaluate(Input,y_fake,verbose=False)

        return fd

    if fd_model == "strong_cate":
        ms = pd.get_dummies(df_fake["marital-status"])
        ms = np.array(pd.DataFrame(ms,columns=['Divorced',
        'Married-AF-spouse',
        'Married-civ-spouse',
        'Married-spouse-absent',
        'Never-married',
        'Separated',
        'Widowed']).fillna(0))

        sex = pd.get_dummies(df_fake["sex"])
        Input = np.concatenate((np.array(ms),np.array(sex)),axis=1)
        Output = np.argmax(np.array(pd.get_dummies(df_fake["relationship"])),axis = 1)
        predicted = np.argmax(model.predict(Input),axis = 1)
        fd = (np.sum(Output != predicted)/y_fake.shape[0] - acc_dict[fd_model]) * 100

        return fd 


    if fd_model == "weak_cate":
        Input = pd.get_dummies(df_fake["education"])
        Input = np.array(pd.DataFrame(Input,columns=['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th',
       'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad',
       'Masters', 'Preschool', 'Prof-school', 'Some-college'],dtype='object').fillna(0))
       
        Output = np.argmax(np.array(pd.get_dummies(df_fake["occupation"])),axis = 1)
        predicted = np.argmax(model.predict(Input),axis = 1) 
        fd = (np.sum(Output != predicted)- acc_dict[fd_model]) * 100

        return fd


def eval_(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    return (mse, mae, mape)


def eval_(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    return (mse, mae, mape)


def sel_loss(x_fake,dataset,sel_train,partition_option, loss_option,fields):
#def sel_loss():
    df_fake = to_df(x_fake,dataset)
    generated = Dataset(
        fields = fields,
        path = None,
        DataFrame = df_fake,
        format = "df")
    generated.learn_convert()
    generated_it = Iterator.split(
            batch_size = 128,
            train = generated)[0]

    generated_data = tf.concat([data for data in generated_it], axis=0)
    generated_data = generated_data.eval(session=tf.compat.v1.Session())

    #generated_data = np.array(tf.concat([data for data in generated_it], axis=0))
    test_data = sel_generation(sel_train,generated_data)

    # print("Test query successfull generated...")
    # #test_data = np.load("dataset/train/adult_converted_sel.npy")


    x_dim = test_data.shape[1]-2
    x_reducedim = x_dim



    tau_part_num = 50

    test_original_X = np.array(test_data[:, :x_dim], dtype=np.float32)
    test_tau_ = []
    for rid in range(test_data.shape[0]):
        t = test_data[rid, x_dim] #hm_to_l2(test_data[rid, x_dim])
        test_tau_.append(t)

    test_tau_ = np.array(test_tau_)
    test_tau = np.zeros((test_data.shape[0], tau_part_num))
    for cid in range(tau_part_num):
        test_tau[:, cid] = test_tau_

    test_Y = np.array(test_data[:, -1], dtype=np.float32)


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
    original_x_dim = test_original_X.shape[1]
    dimreduce_x_dim = x_reducedim



    test_data_predictions_labels_file = os.path.join('./test_face_d128_2M_smallSel_huber_log/', 'test_predictions.npy')
    valid_data_predictions_labels_file = os.path.join('./test_face_d128_2M_smallSel_huber_log/', 'valid_predictions_labels_one_epoch_')

    regression_name = 'adult'
    regression_model_dir = 'pretrained_models/sel'



    regressor = SelNet(hidden_units, vae_hidden_units, batch_size, epochs, epochs_vae,
                            learning_rate, log_option, tau_embedding_size, original_x_dim, dimreduce_x_dim,
                            test_data_predictions_labels_file, valid_data_predictions_labels_file, regression_name, 
                            regression_model_dir, unit_len, max_tau, tau_part_num, partition_option, loss_option)



    predictions = regressor.predict_vae_dnn(test_original_X, test_tau)

    predictions = np.array(predictions)


    # evaluation
    evaluation = eval_(predictions, test_Y)

    return evaluation[1]