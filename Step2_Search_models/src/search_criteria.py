import os
import torch
import numpy as np 
from src.model import CNN_Candidate

def getNumberOfModelParams(model):
	total_params = 0
	for parameter in model.parameters():
		n_params = 1
		for idx, size in enumerate(parameter.shape):
			n_params *= size
		total_params += n_params
	return total_params


def sample_a_CNN(trained_model_ids=[]):
	n_trials = 0
	while(True):
		n_trials=n_trials+1
		# The blocks
		n_blocks = np.random.randint(2,5) # we should have 2~4 blocks
		nn_blocks_arc = []
		for block_id in range(n_blocks):
			block_arc=[]
			block_arc.append(np.random.randint(0,5)) # The first layer can be type 0 --> identity mapping
			block_arc.append(np.random.randint(1,5)) # The second layer must be a convolution module
			block_arc.append(np.random.randint(5,7)) # layer_type 5 and 6 are pooling layers.
			nn_blocks_arc.append(block_arc)

		# The initial filter size (from 8~256)
		filter_size = 8 * np.random.randint(1,33)
		# The growth of n_filters after each block
		filter_growth_coefs = [0.5,0.75,1.0,1.25,1.5,1.75,2.0]
		filter_growth_index = np.random.randint(0,7)
		filter_growth_coef = filter_growth_coefs[filter_growth_index]
		model = CNN_Candidate(nn_blocks_arc, filter_size, filter_growth_coef)

		# Create Model Identifier
		identifier = 'id_'
		for block in nn_blocks_arc:
			for layer_type in block:
				identifier = identifier + str(layer_type)
		identifier = identifier +'_'+ str(filter_size) + '_%.3f'%filter_growth_coef
		# Return the model if it satisfy the size requirement and is brand new.
		model_size = getNumberOfModelParams(model) * 4 / 1024
		if model_size < 500.0 and model_size > 250.0 and (identifier not in trained_model_ids):
			return model, {'identifier': identifier, 'nn_blocks_arc':nn_blocks_arc, 'filter_size': filter_size, 'filter_growth_coef': filter_growth_coef, 'model_size': model_size}
		# If we could not find a satisfactory model...
		if n_trials > 1000:
			print('[ERROR] Tried for 1000 times and no model can be found!!! ')
			return
		



def getLayerTypeString(layer_type_id,in_fil,out_fil):
	if layer_type_id==0:
		return ''
	elif layer_type_id==1:
		return 'conv_dw( %d,  %d, 1)' % (in_fil, out_fil)
	elif layer_type_id==2:
		return 'conv_dw5( %d,  %d, 1)' % (in_fil, out_fil)
	elif layer_type_id==3:
		return 'InvertedResidual(%d, %d, kernel_size=3)' % (in_fil, out_fil)
	elif layer_type_id==4:
		return 'InvertedResidual(%d, %d, kernel_size=5)' % (in_fil, out_fil)
	elif layer_type_id==5:
		return 'nn.AvgPool2d(kernel_size=2)'
	elif layer_type_id==6:
		return 'nn.MaxPool2d(kernel_size=2)'
	else:
		print('Error! Undefined layer type! ')	



def saveModelSpecPy(params_dict, save_folder=None):
	# Read the parameters of model architecture
	nn_blocks_arc = params_dict['nn_blocks_arc']
	filter_size = params_dict['filter_size']
	filter_growth_coef = params_dict['filter_growth_coef']

	if save_folder != None:
		if not os.path.exists(save_folder): os.makedirs(save_folder)
		fo = open(save_folder + '/candidateCNN_struct.py','w')
	else:
		fo = open('candidateCNN_struct.py','w')
	fo.write('import torch \n') 
	fo.write('import numpy as np \n')
	fo.write('import torch.nn as nn \n')
	
	fo.write('class Swish(nn.Module): \n')
	fo.write('	def __init__(self): \n')
	fo.write('		super(Swish, self).__init__() \n')
	fo.write('	def forward(self, x): \n')
	fo.write('		x = x * torch.sigmoid(x) \n')
	fo.write('		return x \n')

	fo.write('class InvertedResidual(nn.Module): \n')
	fo.write('	def __init__(self, inp, oup, kernel_size, stride=1, expand_ratio=3): \n')
	fo.write('		super(InvertedResidual, self).__init__() \n')
	fo.write('		self.kernel_size = kernel_size \n')
	fo.write('		self.padding = (kernel_size - 1) // 2 \n')
	fo.write('		self.stride = stride \n')
	fo.write('		self.use_res_connect = self.stride == 1 and inp == oup \n')
	fo.write('		hidden_dim = int(inp * expand_ratio) \n')
	fo.write('		self.conv = nn.Sequential( \n')
	fo.write('			# pw \n')
	fo.write('			nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), \n')
	fo.write('			nn.BatchNorm2d(hidden_dim), \n')
	fo.write('			Swish(), \n')
	fo.write('			# dw \n')
	fo.write('			nn.Conv2d(hidden_dim, hidden_dim, self.kernel_size, stride, self.padding, groups=hidden_dim, bias=False), \n')
	fo.write('			nn.BatchNorm2d(hidden_dim), \n')
	fo.write('			Swish(), \n')
	fo.write('			# pw-linear \n')
	fo.write('			nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), \n')
	fo.write('			nn.BatchNorm2d(oup), \n')
	fo.write('		) \n')
	fo.write('	def forward(self, x): \n')
	fo.write('		if self.use_res_connect: \n')
	fo.write('			return x + self.conv(x) \n')
	fo.write('		else: \n')
	fo.write('			return self.conv(x) \n')

	fo.write('class CandidateCNN(nn.Module): \n')
	fo.write('	def __init__(self, num_of_classes=3,n_inputChannel=2): \n')
	fo.write('		super(CandidateCNN, self).__init__() \n')
	fo.write('		def conv_bn(inp, oup, stride): \n')
	fo.write('			return nn.Sequential( \n')
	fo.write('				nn.Conv2d(inp, oup, 3, stride, 1, bias=False), \n')
	fo.write('				nn.BatchNorm2d(oup), \n')
	fo.write('				nn.ReLU(inplace=True)) \n')
	fo.write('		def conv_dw(inp, oup, stride): \n')
	fo.write('			return nn.Sequential( \n')
	fo.write('				nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), \n')
	fo.write('				nn.BatchNorm2d(inp), \n')
	fo.write('				Swish(), \n')
	fo.write('				nn.Conv2d(inp, oup, 1, 1, 0, bias=False), \n')
	fo.write('				nn.BatchNorm2d(oup), \n')
	fo.write('				Swish()) \n')
	fo.write('		def conv_dw5(inp, oup, stride): \n')
	fo.write('			return nn.Sequential( \n')
	fo.write('				nn.Conv2d(inp, inp, 5, stride, 1, groups=inp, bias=False), \n')
	fo.write('				nn.BatchNorm2d(inp), \n')
	fo.write('				Swish(), \n')
	fo.write('				nn.Conv2d(inp, oup, 1, 1, 0, bias=False), \n')
	fo.write('				nn.BatchNorm2d(oup), \n')
	fo.write('				Swish()) \n')
	fo.write('		self.model = nn.Sequential( \n')

	fo.write('			conv_bn( n_inputChannel,  ' + str(filter_size) + ', 2),  \n')

	in_fil = filter_size
	out_fil = in_fil
	for block in nn_blocks_arc:
		for i, layer_type_id in enumerate(block):
			if layer_type_id!=0: # Need to expand the number of filters...
				if i==1: out_fil = int(filter_growth_coef * in_fil)
				layer_type_string = getLayerTypeString(layer_type_id,in_fil,out_fil)
				fo.write('			' + layer_type_string + ', \n')
				# Update in_fil
				in_fil=out_fil
	fo.write('			nn.AdaptiveAvgPool2d((1,1)), \n')
	fo.write('		) \n')
	fo.write('		self.fc = nn.Linear(%d, num_of_classes)' % (out_fil) + ' \n')
	fo.write('	def forward(self, x): \n')
	fo.write('		x = self.model(x) \n')
	fo.write('		x = x.view(x.shape[0], -1) \n')
	fo.write('		x = self.fc(x) \n')
	fo.write('		return x \n')
	fo.close()
	return


if __name__ == '__main__':
	model, params_setting = sample_a_CNN()
	print(params_setting)
	torch.save(model.state_dict(), 'modelSizeExample.dict')
	saveModelSpecPy(params_setting)