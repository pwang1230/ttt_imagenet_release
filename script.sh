export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=4 python main.py --shared layer3 --group_norm 32 --workers 16 --outf results/resnet18_layer3_gn
CUDA_VISIBLE_DEVICES=4 python main.py --shared layer3 --workers 16 --outf results/resnet18_layer3_bn

CUDA_VISIBLE_DEVICES=0 python script_test.py --level 5 --shared layer3 --setting online_shuffle --name gn
CUDA_VISIBLE_DEVICES=0 python script_test.py --level 5 --shared layer3 --setting slow --name gn
CUDA_VISIBLE_DEVICES=0 python test_calls/test_video.py --shared layer3 --group_norm 32 --niter 1 \
				--resume results/resnet18_layer3_gn --outf results/triter1_layer3_gn_video

python main.py --shared none --group_norm 32 --workers 16 --outf results/resnet18_layer3_gn
CUDA_VISIBLE_DEVICES=0 python script_test.py --level 5 --shared none --setting none --name gn
