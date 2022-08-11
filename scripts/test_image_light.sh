#!/bin/bash

python Compression_ADMM.py \
		--name="image_compression" \
		--dataroot="./datasets/open_images" \ 
		--gpu_ids=-1 \
		--Color_Input="RGB" \
		--Color_Output="RGB"  \
		--label_nc=3 \
		--output_nc=3  \
		--n_blocks_global=5 \
		--n_downsample_global=5 \
		--image_bit_num=8 \
		--C_channel=64 \
		--n_cluster=64 \
		--max_ngf=256 \
		--resize_or_crop='none' \
		--loadSize=640 \
		--fineSize=640 \
		--Conv_type='C' \
		--quantize_type='scalar' \
		--which_epoch='ADMM_Q_pruned' \
		--fixed_point \
		--ADMM_iter=50  # number of latent updates
