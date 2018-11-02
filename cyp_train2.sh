#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset duke \
--trainset_part trainval \
--exp_dir Exp/20181006_layer3_2LayerMetricLoss_w0.2_duke \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 0.2

CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset duke \
--trainset_part trainval \
--exp_dir Exp/20181006_layer3_2LayerMetricLoss_w0.5_duke \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 0.5

