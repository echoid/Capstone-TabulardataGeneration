import numpy as np
from scipy.spatial import distance
from se
import sys
import math
import pickle
from tqdm import tqdm
import pandas as pd


data = np.load("dataset/train/adult_converted.npy")
query = np.load("dataset/train/adult_converted.npy")
np.random.seed(20)
sc = np.random.choice(data.shape[0], int(data.shape[0]*0.5), replace=False)
query = data[sc]
sel = sel_generation(data,query)