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
					self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
					self.running_var = (1-self.momentum)*self.running_var + self.momentum*var
			
			if self.ttt == True: #Test time domain adaptation
				y = y - self.running_mean.view(-1, 1)
				y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
			else:
				y = y - mean.view(-1,1)
				y = y / (var.view(-1,1)**.5 + self.eps)

		y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
		return y.view(return_shape).transpose(0,1)

class TTTGroupNorm(nn.Module):
	def __init__(self, num_features, num_groups, ttt, eps=1e-5, momentum=0.1, track_running_stats:bool=True):
		super(TTTGroupNorm, self).__init__()
		self.ttt = ttt
		self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
		self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
		self.num_groups = num_groups
		self.eps = eps
		self.running_mean = None
		self.running_var = None
		self.momentum = momentum
		self.track_running_stats = track_running_stats

	def forward(self, x):
		N,C,H,W = x.size()
		G = self.num_groups
		assert C % G == 0

		x = x.view(N,G,-1)
		mean = x.mean(-1, keepdim=True)
		var = x.var(-1, keepdim=True)
		#x = (x-mean) / (var+self.eps).sqrt()
		#x = x.view(N,C,H,W)
		return x * self.weight + self.bias

		#mean = mean.mean(0,keepdim=True)
		#var = var.var(0,keepdim=True)
		if self.training:
			if self.running_mean == None:
				self.running_mean = mean.mean(0,keepdim=True)
				self.running_var = var.var(0,keepdim=True)
			elif self.track_running_stats == True:
				#print(N,C,H,W)
				#print(self.running_mean.shape,mean.shape, self.weight.shape, self.num_groups)
				self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean.mean(0,keepdim=True)
				self.running_var = (1-self.momentum)*self.running_var + self.momentum*var.var(0,keepdim=True)
			
			if self.ttt:
				#Use running mean and var to send forward input
				x = (x-self.running_mean) / (self.running_var+self.eps).sqrt()
				x = x.view(N,C,H,W)
			else:
				#User current mean
				x = (x-mean) / (var+self.eps).sqrt()
				x = x.view(N,C,H,W)
		else:
			#x = (x-mean) / (var+self.eps).sqrt()			
			x = (x-self.running_mean) / (self.running_var+self.eps).sqrt()
			x = x.view(N,C,H,W)
		return x * self.weight + self.bias

