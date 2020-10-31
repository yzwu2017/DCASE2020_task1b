import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


class Swish(nn.Module):
	def __init__(self):
		super(Swish, self).__init__()
	def forward(self, x):
		x = x * torch.sigmoid(x)
		return x





class SeparableConv(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, bias):
		super(SeparableConv, self).__init__()
		padding = (kernel_size - 1) // 2
		self.dwpw = nn.Sequential(
				nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=1, padding=padding, groups=in_planes, bias=bias),
				nn.BatchNorm2d(in_planes),
				Swish(),
				nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias),
				nn.BatchNorm2d(out_planes),
				Swish())
	def forward(self, x):
		out = self.dwpw(x)
		return out


class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, kernel_size, stride=1, expand_ratio=3):
		super(InvertedResidual, self).__init__()
		self.kernel_size = kernel_size
		self.padding = (kernel_size - 1) // 2
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = int(inp * expand_ratio)
		self.use_res_connect = self.stride == 1 and inp == oup

		if expand_ratio == 1:
			self.conv = nn.Sequential(
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, self.kernel_size, stride, self.padding, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim),
				Swish(),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)
		else:
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



class IdentityBranch(nn.Module):
	def __init__(self):
		super(IdentityBranch, self).__init__()
	def forward(self, x):
		return x


def NASLayer(layer_type, in_fil, out_fil):
	if layer_type == 0:
		return IdentityBranch()
	elif layer_type == 1:
		return SeparableConv(in_fil, out_fil, kernel_size=3, bias=False)
	elif layer_type == 2:
		return SeparableConv(in_fil, out_fil, kernel_size=5, bias=False)
	elif layer_type == 3:
		return InvertedResidual(in_fil, out_fil, kernel_size=3)
	elif layer_type == 4:
		return InvertedResidual(in_fil, out_fil, kernel_size=5)
	elif layer_type == 5:
		return torch.nn.AvgPool2d(kernel_size=2)
	elif layer_type == 6:
		return torch.nn.MaxPool2d(kernel_size=2)


# def NASLayer0(layer_type, in_fil, out_fil):
# 	return IdentityBranch()

# def NASLayer1(layer_type, in_fil, out_fil):
# 	return SeparableConv(in_fil, out_fil, kernel_size=3, bias=False)

# def NASLayer2(layer_type, in_fil, out_fil):
# 	return SeparableConv(in_fil, out_fil, kernel_size=5, bias=False)

# def NASLayer3(layer_type, in_fil, out_fil):
# 	return InvertedResidual(in_fil, out_fil, kernel_size=3)

# def NASLayer4(layer_type, in_fil, out_fil):
# 	return InvertedResidual(in_fil, out_fil, kernel_size=5)

# def NASLayer5(layer_type, in_fil, out_fil):
# 	return torch.nn.AvgPool2d(kernel_size=2)

# def NASLayer6(layer_type, in_fil, out_fil):
# 	return torch.nn.MaxPool2d(kernel_size=2)

class CNN_Candidate(nn.Module):
	def __init__(self,
				 nn_blocks_arc=[[0,1,5],[0,1,5]],
				 in_filters=16,
				 filter_growth_coef=2.0,
				 keep_prob=1.0,
				 ):
		super(CNN_Candidate, self).__init__()
		self.in_filters = in_filters
		self.nn_blocks_arc = nn_blocks_arc
		self.keep_prob = keep_prob
		self.num_layers = 3*len(nn_blocks_arc)
		self.stem_conv = nn.Sequential(
			nn.Conv2d(3, self.in_filters, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(self.in_filters, track_running_stats=False))

		self.layers = nn.ModuleList([])
		in_fil = self.in_filters
		out_fil = self.in_filters
		for block_spec in self.nn_blocks_arc:
			for i,layer_type in enumerate(block_spec):
				if i==1:
					out_fil = int(filter_growth_coef*in_fil)
					layer = NASLayer(layer_type, in_fil, out_fil)
					in_fil = out_fil
				else:
					out_fil = in_fil
					layer = NASLayer(layer_type, in_fil, out_fil)
				self.layers.append(layer)

		self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		#self.dropout = nn.Dropout(p=1. - self.keep_prob)
		self.classify = nn.Linear(out_fil, 3)

		# for m in self.modules():
		#	 if isinstance(m, nn.Conv2d):
		#		 nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

	def forward(self, x):
		x = self.stem_conv(x)
		for i in range(self.num_layers):
			pdb.set_trace()
			x = self.layers[i](x)
		x = self.global_avg_pool(x)
		x = x.view(x.shape[0], -1)
		#x = self.dropout(x)
		out = self.classify(x)

		return out


if __name__ == '__main__':
	model = CNN_Candidate()
	x=torch.ones(1,2,128,128)
	y=model(x)
	print(model)
	torch.save(model.state_dict(), 'model.dict')