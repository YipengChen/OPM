#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset cuhk03 \
--trainset_part trainval \
--exp_dir Exp/20181114_original_cuhk03 \
--steps_per_log 20 \
--num_layers 1 \
--weight_metric_loss 0	


CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset cuhk03 \
--trainset_part trainval \
--exp_dir Exp/20181114_layer2_NoMetricLoss_cuhk03 \
--steps_per_log 20 \
--num_layers 2 \
--weight_metric_loss 0	


CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset cuhk03 \
--trainset_part trainval \
--exp_dir Exp/20181114_layer3_NoMetricLoss_cuhk03 \
--steps_per_log 20 \
--num_layers 3 \
--weight_metric_loss 0		


CUDA_VISIBLE_DEVICES=0 python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset cuhk03 \
--trainset_part trainval \
--exp_dir Exp/20181114_layer3_2LayerMetricLoss_w0.2_cuhk03 \
--steps_per_log 20 \
--num_layers 3 \
--weight_metric_loss 0.2	