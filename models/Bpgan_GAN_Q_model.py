import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
#from .Bpgan_VGG_Extractor import Bpgan_VGGLoss
from .networks import VGGFeatureLoss

class Bpgan_GAN_Q_Model(BaseModel):
    def name(self):
        return 'Bpgan_GAN_Q_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc

        ##### define networks        
        # Generator network
        netE_input_nc = input_nc
        self.netE = networks.define_E(input_nc=netE_input_nc,
                                      ngf=opt.ngf,
                                      n_downsample=opt.n_downsample_global,
                                      C_channel=opt.C_channel,
                                      norm=opt.norm,
                                      gpu_ids=self.gpu_ids,
                                      one_D_conv=opt.OneDConv,
                                      one_D_conv_size=opt.OneDConv_size,
                                      max_ngf=opt.max_ngf,
                                      Conv_type=opt.Conv_type,
                                      noisy_activation=opt.noisy_activation, 
                                      fuse_layers=opt.fused_layers)
        self.netDecoder = networks.define_Decoder(output_nc=opt.output_nc,
                                                  ngf=opt.ngf,
                                                  n_downsample=opt.n_downsample_global,
                                                  C_channel=opt.C_channel,
                                                  n_blocks_global=opt.n_blocks_global,
                                                  norm=opt.norm,
                                                  gpu_ids=self.gpu_ids,
                                                  one_D_conv=opt.OneDConv,
                                                  one_D_conv_size=opt.OneDConv_size,
                                                  max_ngf=opt.max_ngf,
                                                  Conv_type=opt.Conv_type,
                                                  Dw_Index=opt.Dw_Index, 
                                                  noisy_activation=opt.noisy_activation, 
                                                  fuse_layers=opt.fused_layers)


        if opt.quantize_type == 'scalar':
            center = torch.arange(start=0, end=opt.n_cluster, step=1.0) / opt.n_cluster
            temp = 1
            self.Q = networks.quantizer(center=center,Temp=temp)
            if len(self.opt.gpu_ids) > 0:
                self.Q = self.Q.cuda()

        elif opt.quantize_type == 'vector':
            center = torch.Tensor(opt.n_cluster,4)
            if len(self.opt.gpu_ids) > 0:
                center = center.cuda()
            temp = 1
            self.Q = networks.vector_quantizer(center=center,Temp=temp)
            if len(self.opt.gpu_ids) > 0:
                self.Q = self.Q.cuda()
        # Discriminator network
        if self.isTrain and not opt.no_gan_loss:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, False, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids,one_D_conv=opt.OneDConv,one_D_conv_size=opt.OneDConv_size)

            
        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)
            self.load_network(self.netDecoder, 'Decoder', opt.which_epoch, pretrained_path)
            self.load_network(self.Q,'Q',opt.which_epoch,pretrained_path)
            if self.isTrain and not opt.no_gan_loss:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)


        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            #self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.MSELoss()
            self.criterion_mse = torch.nn.MSELoss()
            if not opt.no_vgg_loss:
                self.criterionVGG = VGGFeatureLoss(layer='2.2')#Bpgan_VGGLoss()
        
            # Names so we can breakout loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'MSE_Loss', 'Feature', 'D_real', 'D_fake']
            params = list(self.netE.parameters())+list(self.netDecoder.parameters())+list(self.Q.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.9), weight_decay=0)

            if not opt.no_gan_loss:
                # optimizer D                        
                params = list(self.netD.parameters())    
                self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.9))
                self.relu = torch.nn.ReLU()


    def discriminate(self, test_image, use_pool=False):
        input_concat =  test_image.detach()
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def encode_input(self, label_map, real_image=None, infer=False):
        input_label = label_map.data.cuda() if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available() else label_map.data.to("cpu")

        # get edges from instance map

        input_label = Variable(input_label, requires_grad=not infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding

        return input_label, real_image
    def inference(self, label,Q_type = "None", store_code_idx=False):
        # Encode Inputs
        input_label, image  = self.encode_input(Variable(label), infer=True)

        # Fake Generation
        input_concat = input_label
        Compressed_p = self.netE.forward(input_concat)
        Q_Compressed_P = self.Q.forward(Compressed_p,Q_type=Q_type, store_code_idx=store_code_idx)
        fake_image = self.netDecoder.forward(Q_Compressed_P)
        return fake_image,Q_Compressed_P


    def forward(self, label, Q_type = "None"):
        # Encode Inputs
        input_label, _ = self.encode_input(label)

        # Fake Generation

        input_concat = input_label
        Compressed_p = self.netE.forward(input_concat)
        Q_Compressed_p = self.Q.forward(Compressed_p,Q_type=Q_type)
        fake_image = self.netDecoder.forward(Q_Compressed_p)

        return fake_image

    def save(self, which_epoch, save_netD=True):
        self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)
        self.save_network(self.netDecoder, 'Decoder', which_epoch, self.gpu_ids)
        self.save_network(self.Q,'Q',which_epoch,self.gpu_ids)
        if save_netD and not self.opt.no_gan_loss:
            self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netE.parameters())+list(self.netDecoder.parameters())+list(self.Q.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        if not self.opt.no_gan_loss:
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
    def update_learning_rate_custom(self):
        lr = self.opt.lr / 2
        if not self.opt.no_gan_loss:
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

    def update_Temp(self,new_temp):
        self.Q.update_Temp(new_temp)
    def update_center(self,new_center):
        self.Q.update_center(new_center)
