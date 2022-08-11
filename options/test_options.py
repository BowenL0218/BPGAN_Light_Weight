### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=10000, help='how many test images to run')
        self.parser.add_argument('--feature_loss',type=bool,default=True, help='if use feature loss')
        self.parser.add_argument('--input_file',type=str,help="file location for single file test")
        self.parser.add_argument('--center_path', type=str,help="claim the quantization center path")
        self.parser.add_argument('--output_path', type=str,help="output_path")

        # for model compression
        self.parser.add_argument('--num_filters_student', type=int, default=32, help='number of filters of student Encoder')
        self.parser.add_argument('--n_downsample_student', type=int, default=4, help='number of downsampling layers for student Encoder')
        self.parser.add_argument('--norm_student', type=str, default='batch', help='instance normalization, batch normalization, or Identity (no normalization layer)')
        self.parser.add_argument('--Conv_type_student', type=str, default="E", help="C for conventional conv, E for efficient Conv")
        self.parser.add_argument('--latent_lr', type=float, default=0.00625, help='learning rate for latent vector optimization')
        self.parser.add_argument('--ADMM_iter', type=int, default=0, help='number of ADMM iteration steps')
        self.parser.add_argument('--alpha', type=float, default=16, help='alpha for ADMM')
        self.parser.add_argument('--mu', type=float, default=0.001, help='mu for ADMM')
        self.parser.add_argument('--pool_type', type=str, default='none', help='type of pooling layer for encoder. "avg", "max" or "none".')
        self.parser.add_argument('--train_with_ADMM', action='store_true', help='Training with ADMM-optimized latent vector.')

        self.parser.add_argument('--fixed_point', action='store_true', help='8-bit fixed point mode')

        self.isTrain = False
