import torch 
import numpy as np 
import torch.nn as nn 
class Swish(nn.Module): 
	def __init__(self): 
		super(Swish, self).__init__() 
	def forward(self, x): 
		x = x * torch.sigmoid(x) 
		return x 
class InvertedResidual(nn.Module): 
	def __init__(self, inp, oup, kernel_size, stride=1, expand_ratio=3): 
		super(InvertedResidual, self).__init__() 
		self.kernel_size = kernel_size 
		self.padding = (kernel_size - 1) // 2 
		self.stride = stride 
		self.use_res_connect = self.stride == 1 and inp == oup 
		hidden_dim = int(inp * expand_ratio) 
		self.conv = nn.Sequential( 
			# pw 
			nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), 
			nn.BatchNorm2d(hidden_dim), 
			Swish(), 
			# dw 
			nn.Conv2d(hidden_dim, hidden_dim, self.kernel_size, stride, self.padding, groups=hidden_dim, bias=False), 
			nn.BatchNorm2d(hidden_dim), 
			Swish(), 
			# pw-linear 
			nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), 
			nn.BatchNorm2d(oup), 
		) 
	def forward(self, x): 
		if self.use_res_connect: 
			return x + self.conv(x) 
		else: 
			return self.conv(x) 
class CandidateCNN(nn.Module): 
	def __init__(self, num_of_classes=3,n_inputChannel=2): 
		super(CandidateCNN, self).__init__() 
		def conv_bn(inp, oup, stride): 
			return nn.Sequential( 
				nn.Conv2d(inp, oup, 3, stride, 1, bias=False), 
				nn.BatchNorm2d(oup), 
				nn.ReLU(inplace=True)) 
		def conv_dw(inp, oup, stride): 
			return nn.Sequential( 
				nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), 
				nn.BatchNorm2d(inp), 
				Swish(), 
				nn.Conv2d(inp, oup, 1, 1, 0, bias=False), 
				nn.BatchNorm2d(oup), 
				Swish()) 
		def conv_dw5(inp, oup, stride): 
			return nn.Sequential( 
				nn.Conv2d(inp, inp, 5, stride, 1, groups=inp, bias=False), 
				nn.BatchNorm2d(inp), 
				Swish(), 
				nn.Conv2d(inp, oup, 1, 1, 0, bias=False), 
				nn.BatchNorm2d(oup), 
				Swish()) 
		self.model = nn.Sequential( 
			conv_bn( n_inputChannel,  64, 2),  
			InvertedResidual(64, 64, kernel_size=3), 
			nn.AvgPool2d(kernel_size=2), 
			conv_dw( 64,  64, 1), 
			nn.AvgPool2d(kernel_size=2), 
			conv_dw5( 64,  64, 1), 
			conv_dw( 64,  64, 1), 
			nn.AvgPool2d(kernel_size=2), 
			InvertedResidual(64, 64, kernel_size=3), 
			conv_dw( 64,  64, 1), 
			nn.AvgPool2d(kernel_size=2), 
			nn.AdaptiveAvgPool2d((1,1)), 
		) 
		self.fc = nn.Linear(64, num_of_classes) 
	def forward(self, x): 
		x = self.model(x) 
		x = x.view(x.shape[0], -1) 
		x = self.fc(x) 
		return x 
