import numpy as np
from scipy.spatial import distance
import sys
import math
import pickle
from tqdm import tqdm
import pandas as pd


data = np.load("dataset/train/adult_converted.npy")
query = np.load("dataset/train/adult_converted.npy")