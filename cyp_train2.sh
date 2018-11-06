#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset cuhk03 \
--trainset_part trainval \
--exp_dir Exp/20181103_layer2_NoMetricLoss_cuhk03_VGG16Test_24_8_512_NoBN \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 2 \
--weight_metric_loss 0

CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset cuhk03 \
--trainset_part trainval \
--exp_dir Exp/20181103_layer3_NoMetricLoss_cuhk03_VGG16Test_24_8_512_NoBN \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 0

CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset cuhk03 \
--trainset_part trainval \
--exp_dir Exp/20181103_layer3_2LayerMetricLoss_w0.5_cuhk03_VGG16Test_24_8_512_NoBN \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 0.5

CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset cuhk03 \
--trainset_part trainval \
--exp_dir Exp/20181103_layer3_2LayerMetricLoss_w1_cuhk03_VGG16Test_24_8_512_NoBN \
--steps_per_log 20 \
--epochs_per_val 1 \
--num_layers 3 \
--weight_metric_loss 1