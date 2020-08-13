source activate pytorch
export CUDA_VISIBLE_DEVICES=3
nohup python smoothness_clustr.py --dataset cifar10 --base-classifier CIFAR10_resnet18_pretrained_ce_lambda8.0_74per.pth --sigma 0.075 --outfile smoothing_exp/sig_0.075.txt --alpha 0.01 --skip 1 > logs/sig_0.075.log &
