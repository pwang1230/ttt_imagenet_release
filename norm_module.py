import numpy as np
import torch
import torch.nn as nn

class TTTBatchNorm2d(nn.BatchNorm2d):
	def __init__(self, num_features, ttt: bool, eps=1e-5, momentum=0.1, affine:bool=True, \
		track_running_stats:bool=True):
		super(TTTBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
		self.ttt = ttt
		
	def forward(self, x):
		self._check_input_dim(x)
		y = x.transpose(0,1)
		return_shape = y.shape
		y = y.contiguous().view(x.size(1), -1)
		mean = y.mean(dim=1)
		var = y.var(dim=1)
		if self.training == False: #is evaluating
			y = y - self.running_mean.view(-1, 1)
			y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
		else:
			if self.track_running_stats == True:
				with torch.no_grad():
					self.running_mean.copy_((1-self.momentum)*self.running_mean + self.momentum*mean)
					self.running_var.copy_((1-self.momentum)*self.running_var + self.momentum*var)
			
			if self.ttt == True: #Test time domain adaptation
				y = y - self.running_mean.view(-1, 1)
				y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
			else:
				y = y - mean.view(-1,1)
				y = y / (var.view(-1,1)**.5 + self.eps)

		y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
		return y.view(return_shape).transpose(0,1)

class TTTGroupNorm(nn.BatchNorm2d):
	def __init__(self, num_features, num_groups, ttt, eps=1e-5, momentum=0.1, track_running_stats:bool=True, eval_with_rs=False):
		super(TTTGroupNorm, self).__init__(num_features, eps, momentum, True, track_running_stats)
		self.ttt = ttt
		self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
		self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
		self.num_groups = num_groups
		self.eps = eps
		self.iteration_count = 0
		self.eval_with_rs = eval_with_rs
		self.momentum = momentum
		self.track_running_stats = track_running_stats

	def encode_val(self, data, target):
		with torch.no_grad():
			t_dim = target.shape
			l = torch.flatten(target).shape[0]
			l -= torch.flatten(data).shape[0]
			padding = torch.zeros(l).cuda()
			new_data = torch.reshape(torch.cat([torch.flatten(data),padding]),t_dim)
			target.copy_(new_data)

	def extract_val(self,target):
		#extracted into shape (group_norm,1)
		data = torch.flatten(target)
		return torch.reshape(data[:self.num_groups],(self.num_groups,1))

	def forward(self, x):
		N,C,H,W = x.size()
		G = self.num_groups
		assert C % G == 0

		x = x.view(N,G,-1)
		mean = x.mean(-1, keepdim=True)
		var = x.var(-1, keepdim=True)
		if self.training:
			if self.track_running_stats == True:
				running_mean = self.extract_val(self.running_mean)
				running_var = self.extract_val(self.running_var)
				
				self.encode_val((1-self.momentum)*running_mean + self.momentum*mean.mean(0,keepdim=True), self.running_mean)
				self.encode_val((1-self.momentum)*running_var + self.momentum*var.mean(0,keepdim=True), self.running_var)
			if self.ttt:
				#Use running mean and var to send forward input
				running_mean = self.extract_val(self.running_mean)
				running_var = self.extract_val(self.running_var)
				#x = (x-running_mean) / (running_var+self.eps).sqrt()
				self.iteration_count +=1
				alpha = 20
				beta = 0.5
				w_c = (self.iteration_count/alpha)/(self.iteration_count/alpha-np.log(beta))
				w_r = 1-w_c
				
				w_c = 0.99
				w_r = 0.01
				x = (x-(w_r*running_mean+w_c*mean)) / (w_r*running_var+w_c*var+self.eps).sqrt()
				
				x = x.view(N,C,H,W)
			else:
				#User current mean
				x = (x-mean) / (var+self.eps).sqrt()
				x = x.view(N,C,H,W)
		else:
			if self.eval_with_rs == False:
				#evaluate with current mean and var
				x = (x-mean) / (var+self.eps).sqrt()			
			else:
				#evaluate with running stats
				running_mean = self.extract_val(self.running_mean)
				running_var = self.extract_val(self.running_var)
				x = (x-running_mean) / (running_var+self.eps).sqrt()
			x = x.view(N,C,H,W)
		return x * self.weight + self.bias


