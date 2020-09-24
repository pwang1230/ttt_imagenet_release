from subprocess import call
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--shared', default='layer3')
parser.add_argument('--setting', default=None, type=str)
parser.add_argument('--norm_slow_adapt', action='store_true')
parser.add_argument('--norm_momentum', default=0.9, type=float)
parser.add_argument('--name', default=None, type=str)
########################################################################
args = parser.parse_args()

level = args.level
shared = args.shared
setting = args.setting
name = args.name
norm_momentum = args.norm_momentum
norm_slow_adapt = args.norm_slow_adapt

dataroot = '--dataroot '
dataroot += '/proj/vondrick/portia/ImageNet-C/'		# PLEASE EDIT THIS

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
if level == 5:
	common_corruptions.append('original')

none_tag = '--none' if shared == 'none' else ''
if setting == 'slow':
	lr = 0.001
	niter = 10
	online_tag = ''
	shuffle_tag = ''
elif setting == 'online':
	lr = 0.001
	niter = 1
	online_tag = '--online'
	shuffle_tag = ''
elif setting == 'online_shuffle':
	lr = 0.001
	niter = 1
	online_tag = '--online'
	shuffle_tag = '--shuffle'

batch_size_main = 256
batch_size_test = 64

for corruption in common_corruptions:
	print(corruption, 'level', level)
	call(' '.join(['python', 'test_calls/test_initial.py',
						dataroot,
						none_tag,
						'--group_norm	32',
						'--worker		16',
						'--level 		%d' %(level),
						'--corruption	%s' %(corruption),
						'--shared 		%s' %(shared),
						'--batch_size	%d'	%(batch_size_main),
						'--resume 		results/resnet18_%s_%s/' %(shared, name),
						'--outf 		results/test_%s_%s_%s/' %(shared, setting, name)]),
						shell=True)

	if shared == 'none':
		continue

	call(' '.join(['python', 'test_calls/show_decomp.py',
						'--level 		%d' %(level),
						'--corruption	%s' %(corruption),
						'--outf 		results/test_%s_%s_%s/' %(shared, setting, name)]),
						shell=True)

	call(' '.join(['python', 'test_calls/test_adapt.py',
						dataroot,
						online_tag,
						shuffle_tag,
						'--norm_slow_adapt %d' %(norm_slow_adapt),
						'--norm_momentum %f' %(norm_momentum),
						'--group_norm	32',
						'--level 		%d' %(level),
						'--corruption	%s' %(corruption),
						'--shared 		%s' %(shared),
						'--lr 			%f' %(lr),
						'--niter		%d' %(niter),
						'--batch_size	%d'	%(batch_size_test),
						'--resume 		results/resnet18_%s_%s/' %(shared, name),
						'--outf 		results/test_%s_%s_%s/' %(shared, setting, name)]),
						shell=True)

	call(' '.join(['python', 'test_calls/show_result.py',
						'--analyze_avg',
						'--analyze_bin',
						'--analyze_ssh',
						'--level 		%d' %(level),
						'--corruption	%s' %(corruption),
						'--outf 		results/test_%s_%s_%s/' %(shared, setting, name)]),
						shell=True)
