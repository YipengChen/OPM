#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python script/experiment/train_pcb.py \
-d '(1,)' \
--only_test false \
--dataset market1501 \
--trainset_part trainval \
--exp_dir Exp/20181103_layer2_NoMetricLoss_market1501_VGG16Test_24_8_512_NoBN \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 2 \
--weight_metric_loss 0

CUDA_VISIBLE_DEVICES=1 python script/experiment/train_pcb.py \
-d '(1,)' \
--only_test false \
--dataset market1501 \
--trainset_part trainval \
--exp_dir Exp/20181103_layer3_NoMetricLoss_market1501_VGG16Test_24_8_512_NoBN \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 0

CUDA_VISIBLE_DEVICES=1 python script/experiment/train_pcb.py \
-d '(1,)' \
--only_test false \
--dataset market1501 \
--trainset_part trainval \
--exp_dir Exp/20181103_layer3_2LayerMetricLoss_w0.5_market1501_VGG16Test_24_8_512_NoBN \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 0.5

CUDA_VISIBLE_DEVICES=1 python script/experiment/train_pcb.py \
-d '(1,)' \
--only_test false \
--dataset market1501 \
--trainset_part trainval \
--exp_dir Exp/20181103_layer3_2LayerMetricLoss_w1_market1501_VGG16Test_24_8_512_NoBN \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 1