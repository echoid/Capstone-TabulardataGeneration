from data import NumericalField, CategoricalField, Iterator
from data import Dataset
from synthesizer import VGAN_generator, VGAN_discriminator
from synthesizer import V_Train
from util import to_df,KL_Loss,mean_Loss,fd_calculated,sel_loss,sel_net
from random import choice
import os
from selnet import *
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import csv
from tensorflow.python.keras.backend import set_session
warnings.filterwarnings("ignore", category=DeprecationWarning) 





VGAN_variable = {
	"batch_size":[64],
	"z_dim":[128],
	"gen_hidden_dim":[200],
	"gen_num_layers":[3],
	"dis_hidden_dim":[200],
	"dis_num_layers":[3],
	"dis_lstm_dim":[200],
	"lr":[0.0002],
	"noise":[0.1]
}



def parameter_search(Model):
	parameters = {}
	if Model == "VGAN":
		variable = VGAN_variable

	# 创建一个存 parameter的dictionary,随机选择一个parameter
	for param in variable.keys():
		parameters[param] = choice(variable[param])
	return parameters


config = { 
        "name": "adult_fd_test",
		"train": "dataset/train/adult.csv",	
		"sample": "dataset/train/adult.csv",
		"gmm_cols":[],
		"normalize_cols":[0,2,4,10,11,12],
		"one-hot_cols":[1,3,5,6,7,8,9,13,14],
		"ordinal_cols":[],
		"model": "VGAN",
		"n_epochs":20,
		"steps_per_epoch":10,
		"n_search": 5,
		"rand_search": "yes",
		"train_method":"VTrain"
	}


c_dim = 0
condition = False
sample_times = 1
KL = True
GPU = torch.cuda.is_available()

print(GPU)


train = pd.read_csv(config["train"])
n_search = config["n_search"]
search = 1
ratio = 0.9
noise = 0.1
fields = []
col_type = []

# for seletivity
loss_option = 'huber_log'
partition_option = 'l2'


for i, col in enumerate(list(train)):

    if i in config["normalize_cols"]:
        fields.append((col,NumericalField("normalize")))
        col_type.append("normalize")
    elif i in config["gmm_cols"]:
        fields.append((col, NumericalField("gmm", n=5)))
        col_type.append("gmm")
    elif i in config["one-hot_cols"]:
        fields.append((col, CategoricalField("one-hot", noise=noise)))
        col_type.append("one-hot")
    elif i in config["ordinal_cols"]:
        fields.append((col, CategoricalField("dict")))
        col_type.append("ordinal")
    else:
        fields.append((col, CategoricalField("binary",noise=noise)))
        col_type.append("binary")


trn, samp = Dataset.split(
    fields = fields,
    path = ".",
    train = config["train"],
    validation = config["sample"],
    format = "csv",
    valid_ratio=ratio
)

trn.learn_convert()
samp.learn_convert()


model = config["model"]
train_method = config["train_method"]


param = parameter_search(config["model"])


train_it, sample_it = Iterator.split(
    batch_size = param["batch_size"],
    train = trn,
    validation = samp,
    sort_key = None,
    shuffle = True,
    labels = None,
    square = False,
    pad = None
)
x_dim = train_it.data.shape[1]
col_dim = trn.col_dim
col_ind = trn.col_ind


# global data
sel_train = np.load("dataset/train/adult_converted.npy")



def V_Train(t, path, sampleloader, G, D, fd_type,
epochs, lr, dataloader, z_dim, dataset, 
col_type, sample_times, itertimes = 100, 
steps_per_epoch = None, GPU=False, KL=True, method = "ITS",verbose = False):

    # if method != "full":

    #     model_path = "pretrained_models/"
    #     fd_model = tf.keras.models.load_model(model_path + fd_type)
    # else:
    #     fd_model = None
    model_path = "pretrained_models/"
    fd_session = tf.Session()
    fd_graph = tf.get_default_graph()
    set_session(fd_session)
    fd_model = tf.keras.models.load_model(model_path + fd_type)


    sel_model = sel_net(sel_train)


    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    if GPU:
        G.cuda()
        D.cuda()
    G.GPU = True
       
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)

    # the default # of steps is the # of batches.

    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)

    for epoch in range(epochs):
        it = 0
        print("-----------Epoch {}-----------".format(epoch))
        
        while it < steps_per_epoch:
            
            # batch 128, x_real.shape = [128,105]
            for x_real in dataloader:
                if GPU:
                    x_real = x_real.cuda()
                ''' train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                
                if GPU:
                    z = z.cuda()   
                x_fake = G(z)
                
                y_real = D(x_real)
                y_fake = D(x_fake)

                # 生成 0 和 1，避免  Discriminator 对 Generator 压制
                # D_Loss = -(torch.mean(y_real) - torch.mean(y_fake)) # wgan loss
                fake_label = torch.zeros(y_fake.shape[0], 1)
                real_label = np.ones([y_real.shape[0], 1])
                # Avoid the suppress of Discriminator over Generator
                real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
                real_label = torch.from_numpy(real_label).float()
                
                if GPU:

                    fake_label = fake_label.cuda()
                    real_label = real_label.cuda()

                    
                D_Loss1 = F.binary_cross_entropy(y_real, real_label)
                D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)

                D_Loss = D_Loss1 + D_Loss2

                #print("D Loss: {} | fd: {} ".format(D_Loss))
                
                G_optim.zero_grad()
                D_optim.zero_grad()
                D_Loss.backward()
                D_optim.step()

                ''' train Generator '''
                # 生成 一批noisy
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                
                x_fake = G(z)
                y_fake = D(x_fake)

                if (method == "ITS") or (method == "fd") or (method == "full"):
                    # dataset 就是train data
                    
                    df_fake = to_df(x_fake,dataset)
                    
                    G_fd = fd_calculated(df_fake,fd_type,y_fake,fd_model,method,fd_graph,fd_session)
                else:
                    G_fd = 0

                
                

                real_label = torch.ones(y_fake.shape[0], 1)
                if GPU:
                    real_label = real_label.cuda()

                G_origin = F.binary_cross_entropy(y_fake, real_label)

                if (method == "ITS") or (method == "mean") or (method == "full"):
                    G_mean = mean_Loss(x_fake, x_real, col_type, dataset.col_dim)
                else:
                    G_mean = 0

                if KL:
                    G_KL = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)

                else:
                    G_KL = 0

                if (method == "sel") or (method == "full"):
                    
                    G_sel = sel_loss(x_fake,dataset,sel_train,fields,sel_model)
                    

                else:
                    G_sel = 0
                

                G_Loss = G_origin + G_mean + G_fd + G_KL + G_sel

                log = open(path+"train_log"+".txt","a+")
                log.write("{},{},{},{},{},{},{}\n".format(epoch,it,G_origin,G_mean,G_fd,G_KL,G_sel))
                log.close()

                G_optim.zero_grad()
                D_optim.zero_grad()
                G_Loss.backward()
                G_optim.step()

                it += 1

                if verbose:
                    print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))

                if it >= steps_per_epoch:
                    G.eval()
                    for time in range(sample_times):
                        sample_data = None
                        for x_real in sampleloader:
                            z = torch.randn(x_real.shape[0], z_dim)
                            if GPU:
                                z = z.cuda()
                            x_fake = G(z)
                            df = to_df(x_fake,dataset)
                            if sample_data is None:
                                sample_data = df
                            else:
                                sample_data = sample_data.append(df)
                        sample_data.to_csv(path+'sample_data_{}_{}_{}.csv'.format(t,epoch,time), index = None)
                    if GPU:
                        G.cuda()
                        G.GPU = True
                    G.train()
                    break
            #print("G Loss: {} | Origin: {} | KL: {} | fd: {} | mean: {} ".format(G_Loss, G_Loss1, KL_loss, G_fd, G_Loss2))
            if verbose:
                print("Epoch {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
    return G,D




def generation(fd, method):
    # fd == fd types, strong_cate, strong_num, weak_cate, weak_num
    # method == ITS, mean, fd, sel, none
    # path = generated

    path = "dataset/generated/{}/{}/".format(fd,method)

    try:
        os.makedirs(path)
    except:
        pass


    gen = VGAN_generator(param["z_dim"], param["gen_hidden_dim"], x_dim, 
        param["gen_num_layers"], col_type, col_ind, condition=condition,c_dim=c_dim)


    dis = VGAN_discriminator(x_dim, param["dis_hidden_dim"], 
        param["dis_num_layers"],condition,c_dim)


    V_Train(search, path, sample_it, gen, dis, fd,
        1000, param["lr"], train_it, param["z_dim"], trn, col_type, 
        sample_times, itertimes = 100, steps_per_epoch = config["steps_per_epoch"], GPU=GPU, KL=KL, method = method)


# generation("strong_cate","full")
#generation("weak_cate","full")
generation(sys.argv[1],sys.argv[2])
# generation("strong_num","full")
# generation("weak_num","full")


# generation("strong_cate","ITS")
# generation("weak_cate","ITS")
# generation("strong_num","ITS")
# generation("weak_num","ITS")







#sel_loss_2()
#sel_loss()




