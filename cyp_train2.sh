#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset cuhk03 \
--trainset_part trainval \
--exp_dir Exp/20181103_layer1_NoMetricLoss_cuhk03_VGG16Test_24_8_512_NoBN \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 1 \
--weight_metric_loss 0