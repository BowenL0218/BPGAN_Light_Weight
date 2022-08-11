#!/bin/bash
python train_Q.py \
       	--name="open_images" \
	--dataroot="../datasets/open_images" \
	--gpu_ids=0 \
       	--batchSize=16 \
       	--lr=0.0004 \
       	--Color_Input="RGB" \
       	--Color_Output="RGB" \
       	--label_nc=3 \
       	--output_nc=3 \
	--n_blocks_global=5 \
	--n_downsample_global=5 \
	--image_bit_num=8 \
	--C_channel=64 \
	--n_cluster=64 \
       	--max_ngf=256 \
       	--resize_or_crop='crop' \
       	--fineSize=256 \
       	--loadSize=256 \
      	--save_epoch_freq=10 \
       	--niter=150 \
       	--niter_decay=150 \
      	--Conv_type='C' \
       	--Q_train_epoch=100 \
       	--Q_hard_epoch=150  \
       	--nThreads=32 \
      	--lambda_feat=0.008 \
       	--lambda_mse=1. \
      	--beta1=0.9 \
       	--num_D=1 \
       	--no_gan_loss \
       	--no_vgg_loss 
