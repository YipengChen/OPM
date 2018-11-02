#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python script/experiment/train_pcb.py \
-d '(1,)' \
--only_test false \
--dataset duke \
--trainset_part trainval \
--exp_dir Exp/20181006_layer3_2LayerMetricLoss_w2.0_duke \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 2.0

CUDA_VISIBLE_DEVICES=1 python script/experiment/train_pcb.py \
-d '(1,)' \
--only_test false \
--dataset duke \
--trainset_part trainval \
--exp_dir Exp/20181006_layer3_2LayerMetricLoss_w5.0_duke \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 5.0
