#!/bin/bash

python train_Q.py --name="audio_compression" --dataroot="./datasets/timit/timit_mel_8k" --gpu_ids=0 --Color_Input="gray" --Color_Output="gray" --label_nc=1 --output_nc=1 --n_blocks_global=0 --n_downsample_global=3 --image_bit_num=16 --C_channel=4 --n_cluster=16 --sampling_ratio=8000 --resize_or_crop='crop' --no_flip --fineSize=64 --loadSize=90 --batchSize=256  --save_epoch_freq=10 --niter=100 --niter_decay=100 --Q_train_epoch=120 --Q_hard_epoch=150 --Conv_type='E' --quantize_type='scalar' 
