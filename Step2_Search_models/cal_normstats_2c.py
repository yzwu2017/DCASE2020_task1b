import os
import pickle
import numpy as np
import csv
import glob
import json
import pdb


import sklearn.metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from src.functions_asc import *

import matplotlib.pyplot as plt
plt.switch_backend('agg') # Do not show figures.

#===============================
# Loading Config
#===============================
print("Loading Configuration... ")

# Setup the data config
data_config = {	'feat_key': 'feat_scg_avgDiff',
			'file_extension': 'scalogram',
			'development_data_path':'/.../features/DCASE2020_task1b/development/scalogram-J10Q16-avgDiff',
			'developmet_label_doc': '/.../TAU-urban-acoustic-scenes-2020-3class-development/evaluation_setup/fold1_train.csv',
			'FEATURE_DIMENSION':128,
			'BATCH_SIZE':100,
			'SEGMENT_HOP_SIZE':128, # Default is 64
			'SEGMENT_LENGTH':128, # Default is 128
			} 

feat_key = data_config['feat_key']
file_extension = data_config['file_extension']
development_data_path = data_config['development_data_path']
developmet_label_doc = data_config['developmet_label_doc']

FEATURE_DIMENSION = data_config['FEATURE_DIMENSION']
BATCH_SIZE = data_config['BATCH_SIZE']
SEGMENT_HOP_SIZE = data_config['SEGMENT_HOP_SIZE']
SEGMENT_LENGTH = data_config['SEGMENT_LENGTH']


result_folder_name = 'normstats'
normstats_name = 'scalogram-J10Q16-avgDiff.pickle'

#===============================
# Preparing Data
#===============================
train_label_doc = load_label_doc_asc2019(developmet_label_doc)
# Use data loader to load batches of audios.
audioLoader_kwargs = {'num_workers': 1, 'batch_size':300, 'shuffle': True}
train_audio_loader = torch.utils.data.DataLoader(dataset=train_label_doc, **audioLoader_kwargs)

#===============================
# Input Normalization Statistics (!!for input with 3 channels!!)
#===============================
print("Calculating Input Normalization Stats... ")
tic()
# Obtain statistics for input Normalization
#[InputNormMeanGuided11]
# Normalize to [-1,1]. Using mean and according to mean, get the larger diference values from the mean.
# In this setting, train data is in the range of [-1,1], test data is in the range of [?,?].
n_samples = 0
sum_train = np.zeros([2,128])
min_values =  999 * np.ones([2,128])
max_values = -999 * np.ones([2,128])
for idx_audioBatch, (batch_audio_names, _) in enumerate(train_audio_loader):
		# Loading Data.
		X = load_data_3c(batch_audio_names, development_data_path, feat_key, file_extension)
		X = np.array(X)
		sum_train += np.sum(X,axis=(0,2))
		n_samples += X.shape[0] * X.shape[2]
		# Update the maximum and minimum values
		candidate_min_values = np.min(X, axis=(0,2))
		candidate_max_values = np.max(X, axis=(0,2))
		min_values[candidate_min_values < min_values] = candidate_min_values[candidate_min_values < min_values]
		max_values[candidate_max_values > max_values] = candidate_max_values[candidate_max_values > max_values]

mean_train = sum_train / n_samples
std_train_choise1 = mean_train - min_values
std_train_choise2 = max_values - mean_train
std_train_selection = (std_train_choise1 - std_train_choise2) > 0
std_train = std_train_choise2
std_train[std_train_selection] = std_train_choise1[std_train_selection]
# change the format of statistics to match the data
mean_train = mean_train.reshape(1,2,1,FEATURE_DIMENSION)
std_train = std_train.reshape(1,2,1,FEATURE_DIMENSION)
# Save the mean and variance for future testing use.
ensure_folder_exists(result_folder_name)
normalization_stats = {'mean_train': mean_train, 'std_train': std_train}
pickle.dump(normalization_stats, open(result_folder_name + '/' + normstats_name,'wb'))	

toc()