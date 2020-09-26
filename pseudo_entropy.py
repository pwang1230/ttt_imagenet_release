from __future__ import print_function
from __future__ import division
import numpy as np
import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torch.utils.data as torchdata
from torchvision import models as tv_model
from models.ResNet import ResNetCifar as ResNet

parser = argparse.ArgumentParser()
################################################################
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dataset', default='', type=str)

args = parser.parse_args()

cudnn.benchmark = True

# ImageNet based features
classes = 1000
channels = 3
image_size = 224
dataset_root = '/proj/vondrick/portia/ImageNet-C/'
dataset_name = args.dataset

print('==> Building model..')
net = tv_model.resnet18(pretrained=True)

print('==> Preparing datasets..')
val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
torchvision.datasets.ImageFolder(root=dataset_root+dataset_name, transform=val_transform)
loader = torchdata.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
import time
all_epoch_stats = []

def entropy_threshold(args, model, tg_te_loader):
	def entropy(tensor, base=e):
		t = tensor.cpu().detach().numpy()
		num_labels = len(t)
		ent = 0
		for i in range(num_labels):
			ent -= t[i]*log(t[i],base)
		ent = ent / log(num_labels,base)
		return ent

	def accuracy(output, target, topk=(1,5)):
		maxk = max(topk)
		batch_size = target.size(0)
		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))
		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(float(correct_k))
		return res
	
	counter = 0
	ent = []
	acc = []
	model.eval()
		
	print("==> Iterating through classification task and finding entropy...")
	for dictionary in tg_te_loader:
		counter+=1
		inputs = dictionary[0]
		inputs = torch.stack([inputs])[0]
		labels = dictionary[1]
		labels = torch.as_tensor(labels)

		inputs = inputs.cuda()
		labels = labels.cuda()

		outputs = model(inputs)
		softmax = nn.Softmax(1)
		outputs = softmax(outputs)

		for i in range(len(outputs)):
			output = outputs[i]
			en = entropy(output)
			ent.append(en)
			if labels[i]==torch.argmax(output.cpu().detach()):
				acc.append(1)
			else:
				acc.append(0)
	
	print("==> Accuracy per threshold...")
	for i in range(10):
		acc_cls = []
		et = 0.1*(i+1)
		for j in range(len(ent)):
			if ent[j] <= et: #and ent[j] >= entropy_threshold-0.1:
				acc_cls.append(acc[j])
		print("Entropy:", et, str(len(acc_cls))+'/'+str(len(ent)))
		print("\t ==> CLS_acc: "+str(sum(acc_cls)/len(acc_cls)))	
	return 


print('==> Find accuracy for various entropy...')
entropy_threshold(args, model=net, tg_te_loader=loader)
