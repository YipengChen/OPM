#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset market1501 \
--trainset_part trainval \
--exp_dir Exp/20181112_layer3_2LayerMetricLoss_w1.0_market1501 \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 1.0	