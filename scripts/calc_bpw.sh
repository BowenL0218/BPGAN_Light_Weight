#!/bin/bash

python bit_per_weight.py \
                    --name="timit_small_relu4_final" \
                    --checkpoints_dir="./checkpoints" \
                    --dataroot="./datasets/timit/timit_mel_8k" \
                    --gpu_ids=-1 \
                    --Color_Input="gray" \
                    --Color_Output="gray" \
                    --label_nc=1 \
                    --output_nc=1 \
                    --n_blocks_global=0 \
                    --n_downsample_global=3 \
                    --image_bit_num=16 \
                    --C_channel=4 \
                    --n_cluster=16 \
                    --sampling_ratio=8000 \
                    --resize_or_crop='none' \
                    --no_flip \
                    --fineSize=64 \
                    --loadSize=90 \
                    --Conv_type='E' \
                    --quantize_type='scalar' \
                    --which_epoch='ADMM_Q_pruned' \
                    --fixed_point \
                    --fused_layers \
                    --ADMM_iter=0 \
                    --show_act_quant 
