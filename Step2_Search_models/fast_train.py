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
from src.model import CNN_Candidate
from src.search_criteria import sample_a_CNN, saveModelSpecPy

import matplotlib.pyplot as plt
plt.switch_backend('agg') # Do not show figures.


import pandas as pd
# def addRow_dataFrame(df, row):
# 	# df is a dataframe in which we want to add a row.
# 	# row is a list containing the element of each column.
# 	df.loc[-1] = row  # adding a row
# 	df.index = df.index + 1  # shifting index
# 	df = df.sort_index()  # sorting by index
# 	return df

# row = ['112',3000,0.15,0.834,0.910]
def addRow_dataFrame(df, row):
	# df is a dataframe in which we want to add a row.
	# row is a list containing the element of each column.
	df_insert = pd.DataFrame([row],columns=df.columns)
	df = pd.concat([df_insert,df], axis=0) 
	return df


#found_models_list = pd.DataFrame(columns=['identifier', 'model_size', 'loss', 'accuracy', 'seg_accuracy'])
# ensure_folder_exists('found_models')
# found_models_list.to_csv('found_models/found_models_list.csv',index=False)
found_models_list = pd.read_csv('found_models/found_models_list.csv')

_, params_setting = sample_a_CNN()
while params_setting['identifier'] in found_models_list['identifier'].tolist():
	_, params_setting = sample_a_CNN()
else:
	print(params_setting)
	#===============================
	# Configurations
	#===============================
	print("Loading Configuration... ")
	# Read the class spec.
	class_map = np.array(load_csv_file('class_map.csv'))
	num_of_classes = len(class_map)
	className_to_index = {className: int(idx_char) for (idx_char, className) in class_map}
	class_names = class_map[:,1]

	# Setup the data config
	data_config = {	'feat_key': 'feat_scg_avgDiff',
				'file_extension': 'scalogram',
				'development_data_path':'/.../features/DCASE2020_task1b/development/scalogram-J10Q16-avgDiff',
				'developmet_label_doc': '/.../TAU-urban-acoustic-scenes-2020-3class-development/evaluation_setup/fold1_train.csv',
				'evaluation_data_path': '/.../features/DCASE2020_task1b/development/scalogram-J10Q16-avgDiff',
				'evaluation_label_doc': '/.../TAU-urban-acoustic-scenes-2020-3class-development/evaluation_setup/fold1_evaluate.csv',
				'FEATURE_DIMENSION':128,
				'BATCH_SIZE':64,
				'SEGMENT_HOP_SIZE':128, # Default is 64
				'SEGMENT_LENGTH':128, # Default is 128
				'num_of_classes':num_of_classes,
				} 

	feat_key = data_config['feat_key']
	file_extension = data_config['file_extension']
	development_data_path = data_config['development_data_path']
	developmet_label_doc = data_config['developmet_label_doc']
	evaluation_data_path = data_config['evaluation_data_path']
	evaluation_label_doc = data_config['evaluation_label_doc']
	FEATURE_DIMENSION = data_config['FEATURE_DIMENSION']
	BATCH_SIZE = data_config['BATCH_SIZE']
	SEGMENT_HOP_SIZE = data_config['SEGMENT_HOP_SIZE']
	SEGMENT_LENGTH = data_config['SEGMENT_LENGTH']

	print("Initializing... ")
	# Model training configuration
	gpu = 0
	result_folder_name = 'found_models/result-' + params_setting['identifier']
	saveModelSpecPy(params_setting, result_folder_name)
	exec(open(result_folder_name + '/candidateCNN_struct.py').read())
	myModel = CandidateCNN(num_of_classes,2).cuda(gpu)

	LR = 1e-3
	WEIGHT_DECAY_COEFFICIENT = 0.0015
	N_EPOCH = 3

	# Save a initial model.
	ensure_folder_exists(result_folder_name)
	torch.save(myModel.state_dict(), result_folder_name + '/myModel_initial.dict')

	# Serialize data into file:
	json.dump(params_setting, open(result_folder_name+'/model_architecture.txt','w'))
	#params_setting = json.load(open(result_folder_name+'/model_architecture.txt'))

	def data_augmentation(batch_x,batch_y):
		#batch_x = stretch_tensor(batch_x, stretch_scale_HW=(0.2,0), n_patches_HW=(4,1))
		batch_x, batch_y = mixup_samples(batch_x, batch_y, mix_percentage=1.0,gpu=gpu)
		#batch_x = temporal_shift(batch_x, max_shift=127)
		#batch_x = random_erasing(batch_x, min_area=0.02, max_area=0.4, min_aspect_ratio=0.3, erase_percentage=0.5, erased_patch_value=0.0)
		return batch_x, batch_y


	optimizer = torch.optim.Adam(myModel.parameters(), lr=LR, weight_decay=WEIGHT_DECAY_COEFFICIENT)	#optimizer all rnn parameters
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
	#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5, last_epoch=-1)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
	loss_func = nn.BCEWithLogitsLoss().cuda(gpu) 
	sigmoid = nn.Sigmoid()
	#params = list(seg_rnn.parameters()) + list(audio_rnn.parameters()) # if you have two modules to train, concatenate their parameters lists.
	#loss_func = nn.CrossEntropyLoss() #This loss function requires your target to be label number rather than one-hot vector

	# Progress Montioring
	display_step_interval = 80 # After this number of batch training, display the losses.


	#===============================
	# Preparing Data
	#===============================

	train_label_doc = load_label_doc_asc2019(developmet_label_doc)
	eval_label_doc = load_label_doc_asc2019(evaluation_label_doc)

	# Use data loader to load batches of audios.
	audioLoader_kwargs = {'num_workers': 1, 'batch_size':300, 'shuffle': True}
	train_audio_loader = torch.utils.data.DataLoader(dataset=train_label_doc, **audioLoader_kwargs)
	cval_audio_loader = torch.utils.data.DataLoader(dataset=eval_label_doc, batch_size=100, shuffle=False)
	eval_audio_loader = torch.utils.data.DataLoader(dataset=eval_label_doc, batch_size=1, shuffle=False)


	#===============================
	# Input Normalization Statistics (!!for input with 3 channels!!)
	#===============================
	print("Loading Input Normalization Stats... ")
	tic()

	normalization_stats = pickle.load(open('normstats/scalogram-J10Q16-avgDiff.pickle','rb'))    
	mean_train = normalization_stats['mean_train']
	std_train = normalization_stats['std_train']

	toc()
	#===============================
	# Model Training
	#===============================
	print("Model Training... ")

	# Intialize training statistics
	step = 0 
	train_loss=[]
	val_loss=[]
	mean_train = torch.Tensor(mean_train).cuda(gpu)
	std_train = torch.Tensor(std_train).cuda(gpu)

	# Start Training
	for epoch in range(N_EPOCH):
		tic()
		for idx_audioBatch, (batch_audio_names, batch_audio_labels) in enumerate(train_audio_loader):
			# Loading Data.
			#pdb.set_trace()
			X = load_data_3c(batch_audio_names, development_data_path, feat_key, file_extension)
			Y = convert_label_to_nhot(batch_audio_labels, className_to_index)
			X_seg, Y_seg = feature_segmentation_3c(X, Y, SEGMENT_HOP_SIZE, SEGMENT_LENGTH, FEATURE_DIMENSION)
			# Use data loader to load batches of audio segments
			train_seg_loader = torch.utils.data.DataLoader(dataset=list(zip(X_seg,Y_seg)), shuffle=True, batch_size=BATCH_SIZE)
			for idx_segmentBatch, (batch_x, batch_y) in enumerate(train_seg_loader):
				#batch_x = batch_x.unsqueeze(1)
				batch_x = batch_x.float().cuda(gpu)
				batch_y = batch_y.float().cuda(gpu)
				batch_x = (batch_x - mean_train) / std_train
				# Data augmentation
				with torch.no_grad():
					batch_x, batch_y = data_augmentation(batch_x,batch_y)
				# Forward Propagation
				myModel.train()
				output = myModel(batch_x)
				loss = loss_func(output,batch_y)
				# Update the network
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				# Record the number of trained batches 
				step+=1

				# Display the loss values after some batches.
				if step % display_step_interval == 0:
						train_loss.append(float(loss.cpu().data.numpy()))
						validationLoss = compute_avg_loss_of_data_3c(cval_audio_loader,myModel,gpu,loss_func,data_config,className_to_index,mean_train,std_train)
						val_loss.append(validationLoss)
						print('Epoch:%d' % epoch, '| TrainLoss: %.3f' % train_loss[-1], '| ValidationLoss: %.3f' % val_loss[-1])
		#torch.save(myModel.state_dict(), result_folder_name + '/myModel_epoch' + str(epoch) + '.dict')
		scheduler.step()
		toc_train()

	# After training, plot the learning curve.
	plotLearningCurve(train_loss, val_loss, step, result_folder_name)

	# Save the trained model and training curve statistics.
	ensure_folder_exists(os.getcwd() + '/' + result_folder_name)
	torch.save(myModel.state_dict(), result_folder_name + '/myModel.dict')
	#torch.save(myModel, result_folder_name + '/myModel.dict')

	training_curve_stats = {'train_loss': train_loss, 'val_loss': val_loss}
	pickle.dump(training_curve_stats, open(result_folder_name + '/training_curve_stats.pickle','wb'))	


	#==============================================
	# Evaluate the Model Performance on the Test Fold
	#==============================================	

	# Load trained best model
	myModel.load_state_dict(torch.load(result_folder_name + '/myModel.dict', map_location='cuda:' + str(gpu)))

	# 4. Model Evaluation
	print('Evaluating the Model... ')
	tic()
	myModel.eval()
	eval_fnames = []
	eval_labels = []
	eval_preds = []
	eval_labels_seg = [] # Each element in the list is the labels of segments of an audio sample.
	eval_preds_seg = [] # Each element in the list is the predictions of segments of an audio sample.
	for idx_audioSample, [audio_name, audio_label] in enumerate(eval_label_doc):
		# Loading Data.
		#pdb.set_trace()
		X = load_data_3c([audio_name], evaluation_data_path, feat_key, file_extension)
		Y = convert_label_to_nhot([audio_label], className_to_index)
		X_seg, Y_seg = feature_segmentation_3c(X, Y, SEGMENT_HOP_SIZE, SEGMENT_LENGTH, FEATURE_DIMENSION)
		batch_x = torch.Tensor(X_seg).cuda(gpu)
		#batch_x = batch_x.unsqueeze(1)
		batch_x = (batch_x - mean_train) / std_train
		batch_y = np.array(Y_seg)
		with torch.no_grad():
			output = myModel(batch_x)
			output = sigmoid(output)
			# Store the segment-level predictions
			pred_seg = torch.argmax(output,dim=-1)
			eval_preds_seg.append(pred_seg.cpu().numpy())
			label_seg = np.argmax(batch_y,axis=-1)
			eval_labels_seg.append(label_seg)
			# Store the sample-level predictions
			eval_fnames.append(audio_name)
			output_sampleLevel = torch.mean(output,dim=0)
			pred = torch.argmax(output_sampleLevel,dim=-1)
			eval_preds.append(pred.cpu().numpy())
			label = np.argmax(Y[0],axis=-1)
			eval_labels.append(label)

	eval_fnames = np.array(eval_fnames)
	eval_labels = np.array(eval_labels)
	eval_preds = np.array(eval_preds)
	eval_labels_seg =np.array(eval_labels_seg)
	eval_preds_seg = np.array(eval_preds_seg)

	toc()
	#==============================================
	# Result Analysis
	#==============================================	
	print('Analyzing the results... ')

	accuracy_sample = np.sum(eval_labels == eval_preds) / len(eval_labels)
	accuracy_segment = np.sum(eval_labels_seg.reshape(-1) == eval_preds_seg.reshape(-1)) / eval_labels_seg.size
	## Print the overall accuracy 
	print('Model Evaluation: | accuracy: %.3f' % accuracy_sample)
	print('Model Evaluation: | segment-level accuracy: %.3f' % accuracy_segment)



	## Calculate the confusion matrices
	confusion_mtx = getConfusionMatrix(pred_labels=eval_preds, true_labels=eval_labels, n_classes = num_of_classes)
	confusion_mtx_seg = getConfusionMatrix(pred_labels=eval_preds_seg.reshape(-1), true_labels=eval_labels_seg.reshape(-1), n_classes = num_of_classes)

	# Calculate accuracy for each class, and save the results.
	ensure_folder_exists(os.getcwd() + '/' + result_folder_name)
	result_file = open(result_folder_name + '/result_analysis.txt', 'w')
	print('{:25}'.format('Sample-level accuracy') + '{:^.3f}'.format(accuracy_sample), file=result_file)
	print('{:25}'.format('Segment-level accuracy') + '{:^.3f}'.format(accuracy_segment), file=result_file)

	print('-------------------', file=result_file)
	print('Confusion Matrix (Sample): ', file=result_file)
	print(confusion_mtx, file=result_file)

	print('-------------------', file=result_file)
	print('Confusion Matrix (Segment): ', file=result_file)
	print(confusion_mtx_seg, file=result_file)

	print('-------------------', file=result_file)
	print('The change of loss during training: ', file=result_file)
	for idx in range(len(train_loss)):
		print('{:15}'.format('Train Loss: ') + '{:^.3f}'.format(train_loss[idx]) + ' | ' + '{:15}'.format('Val. Loss: ') + '{:^.3f}'.format(val_loss[idx]), file=result_file)

	result_file.close()
		
	# # Plot and save confusion matrix
	# plt.clf()
	# plt.figure(figsize = (10,10))
	# plot_confusion_matrix(confusion_mtx, classes=class_map[:,1], normalize=False, savefig_name = 'cnf_mtx.png', save_folder_name = result_folder_name, 
	# 						title='Confusion matrix (Audio Samples)')
						  

	# plt.clf()
	# plt.figure(figsize = (10,10))
	# plot_confusion_matrix(confusion_mtx_seg, classes=class_map[:,1], normalize=False, savefig_name = 'cnf_mtx_seg.png', save_folder_name = result_folder_name, 
	# 						title='Confusion matrix (Audio Segments)')


	# Update the found model list
	found_models_list = addRow_dataFrame(found_models_list, [params_setting['identifier'], params_setting['model_size'], val_loss[-1], accuracy_sample, accuracy_segment])
	ensure_folder_exists('found_models')
	found_models_list.to_csv('found_models/found_models_list.csv',index=False)
