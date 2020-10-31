import os
import pickle
import numpy as np
import csv
import glob
import pickle
import itertools
import time

import pdb

import random
import cv2
import math

import sklearn.metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F

import matplotlib.pyplot as plt



def compute_avg_loss_of_data(audio_data_loader,model,gpu,loss_func,data_config,className_to_index,mean_train,std_train,batch_size=100):
	model.eval()
	data_folder_path = data_config['evaluation_data_path']
	feat_key = data_config['feat_key']
	file_extension = data_config['file_extension']	
	hop_size = data_config['SEGMENT_HOP_SIZE']
	segment_length = data_config['SEGMENT_LENGTH']
	feature_dimension = data_config['FEATURE_DIMENSION']
	n_steps=0
	avg_loss=0
	sigmoid=nn.Sigmoid()
	for idx_audioBatch, (batch_audio_names, batch_audio_labels) in enumerate(audio_data_loader):
		# Loading Data.
		X = load_data(batch_audio_names, data_folder_path, feat_key, file_extension)
		Y = convert_label_to_nhot(batch_audio_labels, className_to_index)
		X_seg, Y_seg = feature_segmentation(X, Y, hop_size, segment_length, feature_dimension)
		# Use data loader to load batches of audio segments
		seg_loader = torch.utils.data.DataLoader(dataset=list(zip(X_seg,Y_seg)), shuffle=False, batch_size=batch_size)
		for idx_segmentBatch, (batch_x, batch_y) in enumerate(seg_loader):
			with torch.no_grad():
				batch_x = batch_x.unsqueeze(1)
				batch_x = batch_x.float().cuda(gpu)
				batch_y = batch_y.float().cuda(gpu)
				batch_x = (batch_x - mean_train) / std_train
				# Forward Propagation
				output = model(batch_x)
				loss = loss_func(output,batch_y)
				avg_loss += float(loss.cpu().data.numpy())
				n_steps+=1		
	avg_loss = avg_loss / float(n_steps)
	return avg_loss


def compute_avg_loss_of_data_3c(audio_data_loader,model,gpu,loss_func,data_config,className_to_index,mean_train,std_train,batch_size=100):
	model.eval()
	data_folder_path = data_config['evaluation_data_path']
	feat_key = data_config['feat_key']
	file_extension = data_config['file_extension']	
	hop_size = data_config['SEGMENT_HOP_SIZE']
	segment_length = data_config['SEGMENT_LENGTH']
	feature_dimension = data_config['FEATURE_DIMENSION']
	n_steps=0
	avg_loss=0
	sigmoid=nn.Sigmoid()
	for idx_audioBatch, (batch_audio_names, batch_audio_labels) in enumerate(audio_data_loader):
		# Loading Data.
		X = load_data_3c(batch_audio_names, data_folder_path, feat_key, file_extension)
		Y = convert_label_to_nhot(batch_audio_labels, className_to_index)
		X_seg, Y_seg = feature_segmentation_3c(X, Y, hop_size, segment_length, feature_dimension)
		# Use data loader to load batches of audio segments
		seg_loader = torch.utils.data.DataLoader(dataset=list(zip(X_seg,Y_seg)), shuffle=False, batch_size=batch_size)
		for idx_segmentBatch, (batch_x, batch_y) in enumerate(seg_loader):
			with torch.no_grad():
				#batch_x = batch_x.unsqueeze(1)
				batch_x = batch_x.float().cuda(gpu)
				batch_y = batch_y.float().cuda(gpu)
				batch_x = (batch_x - mean_train) / std_train
				# Forward Propagation
				output = model(batch_x)
				loss = loss_func(output,batch_y)
				avg_loss += float(loss.cpu().data.numpy())
				n_steps+=1		
	avg_loss = avg_loss / float(n_steps)
	return avg_loss


def load_csv_file(filename):
	loaded_file = []
	with open(filename) as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace = True)
		for row in csvreader:
			loaded_file.append(row)
	return loaded_file



def load_label_doc_asc2019(label_doc):
	## Read the label document for the data
	with open(label_doc, "r") as text_file:
		lines = text_file.read().split('\n')
	# post-processing the document.
	for idx, ele in enumerate(lines):
		lines[idx]=lines[idx].split('\t')
		lines[idx][0]=lines[idx][0].split('/')[-1].split('.')[0]
	lines = lines[1:] 
	lines = [ele for ele in lines if ele != ['']] # only keep the non-empty elements
	for idx, ele in enumerate(lines):
		lines[idx][-1]=lines[idx][-1].split('\r')[0]
	#label_info=np.array(lines)
	return lines


def load_data(audio_name_list, audio_data_path, feat_key='feat', file_extension='logmel', pad_unit=128, low_energy_value=-80):
	# -------
	# audio_name_list: It is a tuple or list, with each element being the audio feature file name (without extension).
	# pad_unit: If (loaded_feature_length % pad_unit !=0), then pad it with low_energy_value to make (loaded_feature_length % pad_unit == 0)
	# low_energy_value: The value you want to pad. Default is -80.
	# -------
	# load the feature data according to the audio_path_list.
	feat_mtx = []
	for audio_name in audio_name_list:
		audio_path = audio_data_path + '/' + audio_name + '.' + file_extension
		with open(audio_path,'rb') as f:
				temp=pickle.load(f)
				feat_mtx.append(temp[feat_key]) # the loaded feature should be of dimension (n_frames, feature_dimension).
	# pad the audio sequences with vectors with low value (representing silence), to make the sequence length a multiple of pad_unit.
	for idx, feat in enumerate(feat_mtx):
		if feat.shape[0] % pad_unit != 0:
			n_pad = pad_unit - feat.shape[0] % pad_unit
			pad_mtx = low_energy_value * np.ones([n_pad,feat.shape[-1]]) #
			feat_padded = np.concatenate((feat, pad_mtx),axis = 0)
			feat_mtx[idx] = feat_padded
	return feat_mtx

def feature_segmentation(feat_mtx, label_info, hop_size, segment_length, feature_dimension):
	feat_mtx_segmented = []
	label_info_segmented = []
	for idx, feat in enumerate(feat_mtx):
		audio_length = feat.shape[0]
		n_segments = ((audio_length - segment_length) // hop_size) + 1
		audio_segs = np.empty([n_segments,segment_length,feature_dimension])
		# Create segments of data
		for segment_idx in range(n_segments):	#after this loop, file_feat_segments is [num_of_segments, segment_length, feature_dimension]
			seg_data = feat[segment_idx * hop_size: (segment_idx * hop_size) + segment_length, :]
			feat_mtx_segmented.append(seg_data)
		# Create corresponding labels
		for segment_idx in range(n_segments):	#after this loop, file_feat_segments is [num_of_segments, segment_length, feature_dimension]
			label_info_segmented.append(label_info[idx])
	return feat_mtx_segmented, label_info_segmented



def load_data_3c(audio_name_list, audio_data_path, feat_key='feat', file_extension='logmel', pad_unit=128, low_energy_value=-80):
	# -------
	# audio_name_list: It is a tuple or list, with each element being the audio feature file name (without extension).
	# pad_unit: If (loaded_feature_length % pad_unit !=0), then pad it with low_energy_value to make (loaded_feature_length % pad_unit == 0)
	# low_energy_value: The value you want to pad. Default is -80.
	# -------
	# load the feature data according to the audio_path_list.
	feat_mtx = []
	for audio_name in audio_name_list:
		audio_path = audio_data_path + '/' + audio_name + '.' + file_extension
		with open(audio_path,'rb') as f:
				temp=pickle.load(f)
				feat_mtx.append(temp[feat_key]) # the loaded feature should be of dimension (n_frames, feature_dimension).
	# pad the audio sequences with vectors with low value (representing silence), to make the sequence length a multiple of pad_unit.
	for idx, feat in enumerate(feat_mtx):
		if feat.shape[-2] % pad_unit != 0:
			n_pad = pad_unit - feat.shape[-2] % pad_unit
			pad_mtx = low_energy_value * np.ones([feat.shape[0],n_pad,feat.shape[-1]]) # feat.shape[0] is of size n_channels.
			feat_padded = np.concatenate((feat, pad_mtx),axis = -2)
			feat_mtx[idx] = feat_padded
	return feat_mtx

def feature_segmentation_3c(feat_mtx, label_info, hop_size, segment_length, feature_dimension):
	feat_mtx_segmented = []
	label_info_segmented = []
	for idx, feat in enumerate(feat_mtx):
		audio_length = feat.shape[-2]
		n_segments = ((audio_length - segment_length) // hop_size) + 1
		# Create segments of data
		for segment_idx in range(n_segments):	#after this loop, file_feat_segments is [num_of_segments, segment_length, feature_dimension]
			seg_data = feat[:, segment_idx * hop_size: (segment_idx * hop_size) + segment_length, :]
			feat_mtx_segmented.append(seg_data)
		# Create corresponding labels
		for segment_idx in range(n_segments):	#after this loop, file_feat_segments is [num_of_segments, segment_length, feature_dimension]
			label_info_segmented.append(label_info[idx])
	return feat_mtx_segmented, label_info_segmented

def convert_label_to_nhot(audio_labels, className_to_index, ignore_out_of_domain_label=True):
	# Convert label from string to nhont vector. (e.g. "s,m" --> [1,0,0,0,1,0])
	num_of_classes = len(className_to_index)
	audio_labels_nhot=np.zeros([len(audio_labels), num_of_classes])
	for idx,ele in enumerate(audio_labels):
		sample_labels = ele.split(',') # split multiple labels
		if ignore_out_of_domain_label==True:
			for j, class_label in enumerate(sample_labels): # Check each label, if it is not in the specified class name list, ignore it.
				if class_label not in className_to_index: 
					sample_labels = np.delete(sample_labels,j)
					#print('Out-of-domain label \"' + class_label + '\" is ignored.') # For DEBUG Purpose
		sample_labels = [className_to_index[class_label] for class_label in sample_labels] # Convert class names to class indices
		sample_label_nhot = np.sum(np.eye(num_of_classes)[sample_labels], axis=0) # Convert class indices to nhot vector.
		audio_labels_nhot[idx]+=sample_label_nhot
	return audio_labels_nhot


def ensure_folder_exists(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

def plotLearningCurve(train_loss, test_loss, n_steps, result_folder_name='result'):
	# Input:
	#		train_loss: a list contains the training set loss at different time
	#		test_loss: a list contains the validation set loss at different time
	#		n_steps: the total number of training steps
	# Output: None
	if (len(train_loss) != len(test_loss)):
		print("Error when plotting the curve: the train_loss and test_loss should have the same length.")
		return
	num_of_records = len(train_loss)
	title = "Learning Curves"
	plt.title(title)
	plt.xlabel("Training Steps")
	plt.ylabel("Loss")
	plt.grid()
	
	base = np.linspace(0, n_steps, num_of_records)
	plt.plot(base, train_loss, 'o-', color = 'r', label="Training Loss") 
	plt.plot(base, test_loss, 'o-', color = 'b', label="Validation Loss") 
	plt.legend(loc="best")
	plt.show(block=False)
	ensure_folder_exists(result_folder_name)
	plt.savefig(result_folder_name + '/learning_curve.png')



	
def getConfusionMatrix(pred_labels, true_labels, n_classes):
	# the confusion matrix has y-axis being true classes, x-axis being predictions.
	cnf_mtx = np.zeros([n_classes,n_classes])
	for i in range(len(true_labels)):
		cnf_mtx[true_labels[i]][pred_labels[i]] += 1
	cnf_mtx = cnf_mtx.astype('int')
	return cnf_mtx
	
def plot_confusion_matrix(cm, classes,normalize=False, savefig_name = 'cnf_mtx.png', save_folder_name = '', title='Confusion matrix',cmap=plt.cm.Blues):
	# This function prints and plots the confusion matrix.
	# Normalization can be applied by setting `normalize=True`.
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	#plt.show(block=False)
	ensure_folder_exists(os.getcwd() + '/' + save_folder_name)
	plt.savefig(save_folder_name + '/' + savefig_name)
	

# The tic toc functions are originally copied from the following source, though they are modified a little bit.
# Ref: https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python 	
def tic():
	#Homemade version of matlab tic and toc functions
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()
	
def toc():
	if 'startTime_for_tictoc' in globals():
		print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
		del globals()['startTime_for_tictoc']
	else:
		print("Toc: start time not set")


def toc_train():
	if 'startTime_for_tictoc' in globals():
		print("This epoch takes " + str(time.time() - startTime_for_tictoc) + " seconds to finish.")
		del globals()['startTime_for_tictoc']
	else:
		print("Toc: start time not set")




def removeFileExtension(filename):
	string_components = filename.split('.') # there are multiple '.' in the file name.
	if len(string_components) == 2: 
		filename_without_extension = string_components[0]
	elif len(string_components) > 2: 
		filename_without_extension = ''
		for i, strings in enumerate(string_components):
			if (i < len(string_components)-1): filename_without_extension += (strings+'.')
		filename_without_extension=filename_without_extension[:-1]
	return filename_without_extension


def calculateAUC(y_true, y_predict,n_classes):
	auc_for_each_class = np.zeros(n_classes)
	# calculate AUC for each class
	for idx, ele in enumerate(auc_for_each_class):
		auc_for_each_class[idx] = roc_auc_score(y_true = y_true[:,idx], y_score = y_predict[:,idx])
	# calculate the balanced average AUC value (assign higher weights to more prevalent classes)
	n_samples = len(y_true)
	n_positive=np.sum(y_true,axis=0)
	n_negative= n_samples - n_positive
	classes_prevalence = n_positive / np.sum(n_positive)
	balanced_AUC = np.sum(classes_prevalence*auc_for_each_class)
	# calculate the unbalanced average AUC value (assign all class with equal weights)
	unbalanced_AUC = np.mean(auc_for_each_class)
	return balanced_AUC, unbalanced_AUC, auc_for_each_class

# It seems there exist multiple definition of mean average precision. This one is defined as the area under the precision-recall curve.
def calculateMAP(y_true, y_predict,n_classes):
	meanAP_for_each_class = np.zeros(n_classes)
	# calculate MAP for each class
	for idx, ele in enumerate(meanAP_for_each_class):
		meanAP_for_each_class[idx] = average_precision_score(y_true = y_true[:,idx], y_score = y_predict[:,idx])
	# calculate the balanced MAP value (assign higher weights to more prevalent classes)
	n_samples = len(y_true)
	n_positive=np.sum(y_true,axis=0)
	n_negative= n_samples - n_positive
	classes_prevalence = n_positive / np.sum(n_positive)
	weighted_MAP = np.sum(classes_prevalence*meanAP_for_each_class)
	# calculate the unbalanced MAP value (assign all class with equal weights)
	mean_AP = np.mean(meanAP_for_each_class)
	return weighted_MAP, mean_AP, meanAP_for_each_class


# def apk(actual, predicted, k=10):
#     """
#     Computes the average precision at k.
#     This function computes the average prescision at k between two lists of
#     items.
#     Parameters
#     ----------
#     actual : list
#              A list of elements that are to be predicted (order doesn't matter)
#     predicted : list
#                 A list of predicted elements (order does matter)
#     k : int, optional
#         The maximum number of predicted elements
#     Returns
#     -------
#     score : double
#             The average precision at k over the input lists
#     """
#     if len(predicted)>k:
#         predicted = predicted[:k]
#     score = 0.0
#     num_hits = 0.0
#     for i,p in enumerate(predicted):
#         if p in actual and p not in predicted[:i]:
#             num_hits += 1.0
#             score += num_hits / (i+1.0)
#     if not actual:
#         return 0.0
#     return score / min(len(actual), k)



def save_a_ROC_Curve(fpr, tpr, class_name='',result_folder_name='result'):
	# Plot and save the ROC curve for a specific class.
	plt.clf()
	title = "ROC Curve for class \"" + class_name + "\""
	plt.title(title)
	plt.xlabel("FP rate")
	plt.ylabel("TP rate")
	plt.grid()
	plt.plot(fpr, tpr, '-', color = 'b',label='') 
	#plt.plot(fpr, tpr, '-', color = 'b', label=class_name) 
	plt.legend(loc="best")
	ensure_folder_exists(result_folder_name)
	plt.savefig(result_folder_name + '/roc_curve_for_class_' + class_name +'.png')



def cal_ROC_curve_stats(y_trues, y_scores, num_of_classes):
	fprs=[]
	tprs=[]
	thrds=[]
	for idx in range(num_of_classes):
		fpr,tpr,thresholds = sklearn.metrics.roc_curve(y_true=y_trues[:,idx], y_score=y_scores[:,idx])
		fprs.append(fpr)
		tprs.append(tpr)
		thrds.append(thresholds)
	return fprs,tprs,thrds

def save_ROC_Curves(fprs, tprs, class_names, result_folder_name='result'):
	# Plot and save the ROC curve for all classes.
	plt.clf()
	title = "ROC Curves"
	plt.title(title)
	plt.xlabel("FP rate")
	plt.ylabel("TP rate")
	plt.grid()
	for idx,cname in enumerate(class_names):
		plt.plot(fprs[idx], tprs[idx], '-', label=cname) 
	plt.legend(loc="best")
	ensure_folder_exists(result_folder_name)
	plt.savefig(result_folder_name + '/roc_curves.png')
	return

def cal_PR_curve_stats(y_trues, y_scores, num_of_classes):
	precisions=[]
	recalls=[]
	thrds=[]
	for idx in range(num_of_classes):
		p,r,thresholds = sklearn.metrics.precision_recall_curve(y_true=y_trues[:,idx],probas_pred=y_scores[:,idx])
		precisions.append(p)
		recalls.append(r)
		thrds.append(thresholds)
	return precisions,recalls,thrds


def save_PR_Curves(precisions, recalls, class_names, result_folder_name='result'):
	# Plot and save the Precision-Recall curve for all classes.
	plt.clf()
	title = "Precision-Recall Curves"
	plt.title(title)
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.grid()
	for idx,cname in enumerate(class_names):
		plt.plot(recalls[idx], precisions[idx], '-', label=cname) 
	plt.legend(loc="best")
	ensure_folder_exists(result_folder_name)
	plt.savefig(result_folder_name + '/pr_curves.png')
	return


def get_sequential_label_predictions(audio_sequential_predictions,class_thresholds,class_names):
	label_predictions=[]
	predictions_nhot = audio_sequential_predictions>class_thresholds
	for nhot in predictions_nhot:
		labels = class_names[nhot]
		# save the final label to label_predictions
		label_predictions.append(','.join(labels))
	return label_predictions



#======================================
# For generating transcription from unlabeled data.
#======================================

def feature_segmentation_dataOnly(feat_mtx, hop_size, segment_length, feature_dimension):
	# Segmentation of feature data without labels.
	feat_mtx_segmented = []
	for idx, feat in enumerate(feat_mtx):
		audio_length = feat.shape[0]
		n_segments = ((audio_length - segment_length) // hop_size) + 1
		audio_segs = np.empty([n_segments,segment_length,feature_dimension])
		# Create segments of data
		for segment_idx in range(n_segments):	#after this loop, file_feat_segments is [num_of_segments, segment_length, feature_dimension]
			seg_data = feat[segment_idx * hop_size: (segment_idx * hop_size) + segment_length, :]
			feat_mtx_segmented.append(seg_data)
	return feat_mtx_segmented


def get_audio_sample_lists(dataset_path_list_file, feature_type):
	# From a dataset_path_list file, obtain the unlabeled data samaples' path list.
	# ---DEBUG---
	# dataset_path_list_file = 'audio_to_cut.list'
	# feature_type = 'logmel-64'
	# -----------
	# Read the dataset_path_list
	dataset_path_list=[]
	f = open(dataset_path_list_file,'r')
	for line in f:
		dataset_path = line.split('\n')[0]
		dataset_path = dataset_path.split('\r')[0]
		dataset_path_list.append(dataset_path)
	# get dataset names and the corresponding data sample list
	dataset_names = [ele.split('/')[-1] for ele in dataset_path_list] # The folder name is the dataset's name.
	audio_lists = [glob.glob(dataset_path + '/' + feature_type + '/*.pickle') for dataset_path in dataset_path_list]
	return dataset_names, audio_lists


#======================================
# For generating grad-cam visualization
#======================================

def get_label_docs(dataset_path_list_file, feature_type, className_to_index):
	# ---DEBUG---
	# dataset_path_list_file = 'train_data.list'
	# feature_type = 'logmel-64'
	# className_to_index
	# -----------
	num_of_classes = len(className_to_index)
	# Read the dataset_path_list
	dataset_path_list=[]
	f = open(dataset_path_list_file,'r')
	for line in f:
		dataset_path = line.split('\n')[0]
		dataset_path = dataset_path.split('\r')[0]
		dataset_path_list.append(dataset_path)
	# get the dataset label files' path
	dataset_names = [ele.split('/')[-1] for ele in dataset_path_list] # The folder name is the dataset's name.
	dataset_label_doc_list = [dataset_path+'.csv' for dataset_path in dataset_path_list] # In each dataset, there should be a label file with exactly the same name as the dataset's name.
	# load label_doc for each dataset.
	label_docs_raw = []
	for idx, label_doc_path in enumerate(dataset_label_doc_list):
		label_doc = load_csv_file(label_doc_path)
		label_doc = [[dataset_path_list[idx]+'/'+feature_type+'/'+ele[0], ele[1]] for ele in label_doc]
		label_docs_raw.append(label_doc)
	# Process the label_docs_raw to obtain label_docs_final, which is a list of audio_list for each dataset, each row is like [full_path_to_the_feature, feature_label_in_nhot]
	label_docs_final = []
	for _, audio_list in enumerate(label_docs_raw):
		label_doc_final = []
		for ele in audio_list: # each row in audio list contains an audio sample's path and its label.
			sample_raw_path = ele[0].split('.') # there are multiple '.' in the file name.
			sample_full_path = ''
			if len(sample_raw_path) == 2: # the string after the last '.' is the file extension, change it to the feature_file_extension
				sample_full_path += sample_raw_path[0] + '.pickle'
			elif len(sample_raw_path) > 2: # If the file name contain duration like "3.5s", then there exist multiple '.'
				for i, strings in enumerate(sample_raw_path):
					if (i < len(sample_raw_path)-1): sample_full_path += (strings+'.')
				sample_full_path += 'pickle'
			else:
				print(sample_raw_path)
			sample_labels = ele[1]
			# sample_labels = ele[1].split(',') # split multiple labels
			# sample_labels = [className_to_index[class_label] for class_label in sample_labels] # Convert class names to class indices
			# sample_label_nhot = np.sum(np.eye(num_of_classes)[sample_labels], axis=0) # Convert class indices to nhot vector.
			label_doc_final.append([sample_full_path,sample_labels])
		label_docs_final.append(label_doc_final)
	return dataset_names, label_docs_final



#======================
# Data Augmentation
#======================

def background_subtraction(feat_mtx):
	for sample_idx, feat in enumerate(feat_mtx):
		mean_of_feat = np.mean(feat)
		feat_mtx[sample_idx] = feat - mean_of_feat
	return feat_mtx

def stretch_tensor(tensor_in, stretch_scale_HW=(0.45,0.45), n_patches_HW=(4,4)):
	# this function can be used to do the time-frequency stretching of 4D tensor.
	image_H = tensor_in.size(2)
	image_W = tensor_in.size(3)
	n_patches_H = n_patches_HW[0]
	n_patches_W = n_patches_HW[1]
	stretch_scale_H = stretch_scale_HW[0]
	stretch_scale_W = stretch_scale_HW[1]
	# The stretched coords are derived from the original grid coordinates.
	mean_patch_H = image_H / n_patches_H
	mean_patch_W = image_W / n_patches_W
	grid_coords_H = np.linspace(1,image_H,n_patches_H+1,dtype='int')
	grid_coords_W = np.linspace(1,image_W,n_patches_W+1,dtype='int')
	# Make the stretching coordinates for H-axis
	stretch_coef_H = 2*np.random.rand(len(grid_coords_H))-1 # generate random number of range [-1,1)
	stretch_coef_H[[0,-1]] = 0
	stretch_pixels_H = mean_patch_H * stretch_scale_H * stretch_coef_H
	stretch_pixels_H = stretch_pixels_H.astype(int) 
	stretched_coords_H = grid_coords_H + stretch_pixels_H
	# Make the stretching coordinates for W-axis
	stretch_coef_W = 2*np.random.rand(len(grid_coords_W))-1 # generate random number of range [-1,1)
	stretch_coef_W[[0,-1]] = 0
	stretch_pixels_W = mean_patch_W * stretch_scale_W * stretch_coef_W
	stretch_pixels_W = stretch_pixels_W.astype(int) 
	stretched_coords_W = grid_coords_W + stretch_pixels_W
	# Creating the stretched tensor.
	tensor_out = torch.zeros(tensor_in.shape)
	for i in range(n_patches_H):
		for j in range(n_patches_W):
			patch = tensor_in[:,:,i*mean_patch_H:(i+1)*mean_patch_H,j*mean_patch_W:(j+1)*mean_patch_W]
			stretch_shape = (stretched_coords_H[i+1]-stretched_coords_H[i],stretched_coords_W[j+1]-stretched_coords_W[j])
			stretched_patch = F.upsample(patch, size=stretch_shape, mode='bilinear')
			tensor_out[:,:,stretched_coords_H[i]:stretched_coords_H[i+1],stretched_coords_W[j]:stretched_coords_W[j+1]] = stretched_patch
	# tensor_out is the stretched tensor.
	return tensor_out


# # [For Debug] Check the beta distribution
# r = np.random.beta(2.0, 2.0, size=1000)
# plt.hist(r) #histtype='stepfilled', alpha=0.2)
# plt.show()

# [For Debug]
# batch_data = torch.Tensor(np.random.random([5,1,3,2]))
# batch_labels = torch.eye(5)
def mixup_samples(batch_data, batch_labels, mix_percentage=1.0, gpu=None):
	# Note: In the original paper https://arxiv.org/pdf/1710.09412.pdf, the weights follow beta distribution np.random.beta(alpha, alpha,len(batch_data)), we may set alpha=0.2. 
	# Though considering the speed of sampling, uniform distribution is faster.
	mix_weights_label =  torch.Tensor(np.random.uniform(low=0.0, high=1.0, size=len(batch_data))).unsqueeze(1) 
	if gpu!=None:
		mix_weights_label=mix_weights_label.cuda(gpu)
	#mix_weights_label =  torch.Tensor(np.random.beta(0.2, 0.2, size=len(batch_data))).unsqueeze(1) 
	mix_weights_data =mix_weights_label.unsqueeze(1).unsqueeze(1)
	exotic_sample_indices = np.arange(len(batch_data))
	np.random.shuffle(exotic_sample_indices) # Shuffling the indices so that a random sample is chosen as the exotic sample.
	batch_data_mixed = (1-mix_weights_data) * batch_data + mix_weights_data * batch_data[exotic_sample_indices]
	batch_labels_mixed = (1-mix_weights_label) * batch_labels + mix_weights_label * batch_labels[exotic_sample_indices]
	if mix_percentage > 0.0 and mix_percentage < 1.0:
		n_mix = int(mix_percentage * len(batch_data))
		batch_data_mixed[n_mix:] = batch_data[n_mix:]
		batch_labels_mixed[n_mix:] = batch_labels[n_mix:]
	return batch_data_mixed, batch_labels_mixed





# [For Debug]
# import matplotlib.pyplot as plt
# batch_data = torch.Tensor(np.ones([5,1,10,10]))
# batch_data = random_erasing(batch_data)
# plt.imshow(batch_data[0,0])
# plt.show()
# reference: https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
# [Comments]: 
# Now the implementation is like erasing the same patch in a batch of samples, While it seems most reasonable to do it for each image separately.
# Not quite sure about the influence.
def random_erasing(batch_data, min_area=0.02, max_area=0.4, min_aspect_ratio=0.3, erase_percentage=0.5, erased_patch_value=0.0):
	for attempt in range(100): # If tried for 100 times and failed, just return the original batch data.
		area = batch_data.shape[2] * batch_data.shape[3]
		target_area = random.uniform(min_area, max_area) * area
		aspect_ratio = random.uniform(min_aspect_ratio, 1/min_aspect_ratio)
		h = int(round(math.sqrt(target_area * aspect_ratio)))
		w = int(round(math.sqrt(target_area / aspect_ratio)))	
		if h < batch_data.shape[2] and w < batch_data.shape[3]:
			x1 = random.randint(0, batch_data.shape[2] - h)
			y1 = random.randint(0, batch_data.shape[3] - w)
			batch_data[:, :, x1:x1+h, y1:y1+w] = erased_patch_value
			return batch_data
	return batch_data




def temporal_shift(batch_data, max_shift=127):
	shift_value = np.random.randint(low=0, high=127)
	batch_data = batch_data.roll(shifts=shift_value, dims=2)
	return torch.Tensor(batch_data)

