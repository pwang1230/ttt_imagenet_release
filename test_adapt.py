from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from utils.misc import *
from utils.adapt_helpers import *

parser = argparse.ArgumentParser()
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--dataroot', default='/proj/vondrick/datasets/ImageNet-ILSVRC2012/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--csv_path', default='ILSVRC-100_layer_0_comp0_all_100%.csv', type=str)
parser.add_argument('--eval_with_rs', action='store_true')
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--norm_slow_adapt', default=0, type=int)
parser.add_argument('--norm_momentum',type=float)
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=10, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--dset_size', default=0, type=int)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='.')

args = parser.parse_args()
args.csv_root = '/proj/vondrick/portia/Novelty_ImageNet/results/ILSVRC_csv/'
args.csv_path = '/proj/vondrick/portia/Novelty_ImageNet/results/ILSVRC_csv/val/'+args.csv_path

args.threshold += 0.001		# to correct for numeric errors
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)
teset, _ = prepare_test_data(args, use_transforms=False)

print('Resuming from %s...' %(args.resume))
ckpt = torch.load('%s/ckpt.pth' %(args.resume))
if args.online:
	net.load_state_dict(ckpt['net'])
	head.load_state_dict(ckpt['head'])

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(ssh.parameters(), lr=args.lr)

print('Running...')
teset.switch_mode(True, False)
if args.dset_size > 0:
	teset, _ = torch.utils.data.random_split(teset, (args.dset_size, len(teset)-args.dset_size))
if args.shuffle:
	teset, _ = torch.utils.data.random_split(teset, (len(teset), 0))
correct = []
sshconf = []
trerror = []
for i in tqdm(range(1, len(teset)+1)):
	if not args.online:
		net.load_state_dict(ckpt['net'])
		head.load_state_dict(ckpt['head'])

	image, label = teset[i-1]
	sshconf.append(test_single(ssh, image, 0)[1])
	if sshconf[-1] < args.threshold:
		adapt_single(ssh, image, optimizer, criterion, args.niter, args.batch_size)
	correct.append(test_single(net, image, label)[0])
	trerror.append(trerr_single(ssh, image))
	if len(correct) % 100 == 0:
		print(sum(correct)/len(correct),'(',sum(correct[len(correct)-100:])/100,')')

print('Adapted test error cls %.2f' %((1-mean(correct))*100))
rdict = {'cls_correct': np.asarray(correct), 'ssh_confide': np.asarray(sshconf), 
		'cls_adapted':1-mean(correct), 'trerror': trerror}
torch.save(rdict, args.outf + '/%s_%d_ada.pth' %(args.corruption, args.level))
