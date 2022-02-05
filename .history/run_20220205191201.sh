#!/bin/bash

data=$1

path = "dataset/train"

data_file='../../data/'${data}'/real_data/'${data}'_originalData.npy'
train_feats_file='./feats/'${data}'_trainingFeats.npy'
valid_feats_file='./feats/'${data}'_validationFeats.npy'
test_feats_file='./feats/'${data}'_testingFeats.npy'
mkdir './feats'



python selectivity_generation.py ${data_file} ${train_feats_file} ${train_result_file} ${train_data_file}

