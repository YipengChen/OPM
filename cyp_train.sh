#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python script/experiment/train_pcb.py \
-d '(1,)' \
--only_test false \
--dataset market1501 \
--trainset_part trainval \
--exp_dir Exp/20181112_original_market1501 \
--steps_per_log 20 \
--num_layers 1 \
--weight_metric_loss 0	


CUDA_VISIBLE_DEVICES=1 python script/experiment/train_pcb.py \
-d '(1,)' \
--only_test false \
--dataset market1501 \
--trainset_part trainval \
--exp_dir Exp/20181112_layer2_NoMetricLoss_market1501 \
--steps_per_log 20 \
--num_layers 2 \
--weight_metric_loss 0	


CUDA_VISIBLE_DEVICES=1 python script/experiment/train_pcb.py \
-d '(1,)' \
--only_test false \
--dataset market1501 \
--trainset_part trainval \
--exp_dir Exp/20181112_layer3_NoMetricLoss_market1501 \
--steps_per_log 20 \
--num_layers 3 \
--weight_metric_loss 0		


CUDA_VISIBLE_DEVICES=1 python script/experiment/train_pcb.py \
-d '(1,)' \
--only_test false \
--dataset market1501 \
--trainset_part trainval \
--exp_dir Exp/20181112_layer3_2LayerMetricLoss_w0.2_market1501 \
--steps_per_log 20 \
--num_layers 3 \
--weight_metric_loss 0.2	