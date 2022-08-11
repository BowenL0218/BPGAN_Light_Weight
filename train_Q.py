### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_Q_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.utils.tensorboard import  SummaryWriter
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
intial_flag = True
opt.model = 'Bpgan_GAN_Q'
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    Temp = (opt.Q_final - opt.Q_init_Temp)/(opt.Q_hard_epoch-opt.Q_train_epoch) *(start_epoch-opt.Q_train_epoch)+opt.Q_init_Temp
    if start_epoch >= opt.Q_train_epoch:
        intial_flag = False
else:
    start_epoch, epoch_iter = 1, 0
if opt.debug:
    opt.batchSize=2
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 100
    opt.niter_decay = 50
    opt.max_dataset_size = 10
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
summary_path = os.path.join(opt.checkpoints_dir,opt.name,'logs/')
writter = SummaryWriter(log_dir=summary_path)
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
random_index = 20

imtype = np.uint8 if opt.image_bit_num==8 else np.uint16


def train(data, epoch, total_steps, epoch_iter, Q_type):
    iter_start_time = time.time()
    total_steps += opt.batchSize
    epoch_iter += opt.batchSize

    # whether to collect output images
    save_fake = total_steps % opt.display_freq == 0

    ############## Forward Pass ######################
    fake_image = model(Variable(data['label']), Q_type=Q_type)
    real_image = data['image'].cuda()

    loss_D_fake = 0
    loss_D_real = 0
    loss_G_GAN = 0

    model.module.optimizer_G.zero_grad()
    criterionGAN = nn.BCEWithLogitsLoss()
    if not opt.no_gan_loss:
        model.module.optimizer_D.zero_grad()

        # Fake Detection and Loss
        pred_fake_pool = model.module.discriminate(fake_image)[0][0]
        #loss_D_fake = opt.lambda_gan * F.relu(1.0 + pred_fake_pool).mean()#self.criterionGAN(pred_fake_pool, False)        
        loss_D_fake = opt.lambda_gan * criterionGAN(pred_fake_pool, torch.zeros_like(pred_fake_pool))

        # Real Detection and Loss        
        pred_real = model.module.discriminate(real_image)[0][0]
        #loss_D_real = opt.lambda_gan * F.relu(1.0 - pred_real).mean()
        loss_D_real = opt.lambda_gan * criterionGAN(pred_real, torch.ones_like(pred_real))

        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()

        model.module.optimizer_D.step()

        # GAN loss (Fake Passability Loss)        
        pred_fake = model.module.netD.forward(fake_image)[0][0]
        #loss_G_GAN = - opt.lambda_gan * pred_fake.mean()
        loss_G_GAN = opt.lambda_gan * criterionGAN(pred_fake, torch.ones_like(pred_fake))
        
    # GAN feature matching loss
    loss_G_GAN_Feat = 0
    if not opt.no_gan_loss and not opt.no_ganFeat_loss:
        pass
        """
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(self.opt.num_D):
            for j in range(len(pred_fake[i])-1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                    self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        """
            
    # VGG feature matching loss
    loss_G_VGG = 0
    if not opt.no_vgg_loss:
        loss_G_VGG = model.module.criterionVGG(fake_image, real_image) * opt.lambda_feat
    loss_mse = 0
    if not opt.no_mse_loss:
        loss_mse = model.module.criterion_mse(fake_image,real_image) * opt.lambda_mse
    

    loss_G = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG + loss_mse
    
    ############### Backward Pass ####################
    # update generator weights
    loss_G.backward()
    model.module.optimizer_G.step()

    loss_dict = {'D_fake': loss_D_fake,
                    'D_real': loss_D_real,
                    'G_GAN': loss_G_GAN,
                    'G_GAN_Feat': loss_G_GAN_Feat,
                    'Feature': loss_G_VGG,
                    'MSE_Loss': loss_mse
                    }

    ############## Display results and errors ##########
    ### print out errors
    if total_steps % opt.print_freq == 0:
        errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
        t = (time.time() - iter_start_time) / opt.batchSize
        visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        #visualizer.plot_current_errors(errors, total_steps)
    ### display output images
    if save_fake:
        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], 0, imtype)),
                                ('synthesized_image', util.tensor2im(fake_image.data[0], imtype))])
        visualizer.display_current_results(visuals, epoch, total_steps)

    ### save latest model
    if total_steps % opt.save_latest_freq == 0:
        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
        model.module.save('latest')
        np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

    return total_steps, epoch_iter


for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    if epoch < opt.Q_train_epoch:
        for i, data in enumerate(dataset, start=epoch_iter):
            total_steps, epoch_iter = train(data, epoch, total_steps, epoch_iter, "None")

    elif epoch >= opt.Q_train_epoch and epoch < opt.Q_hard_epoch:
        if epoch == opt.Q_train_epoch:
            model.module.save('floating_final')
        if intial_flag:
            intial_flag = False

            center = torch.linspace(0, 1, opt.n_cluster).cuda()
            model.module.update_center(center)
        model.module.netE.train()
        for i, data in enumerate(dataset, start=epoch_iter):
            total_steps, epoch_iter = train(data, epoch, total_steps, epoch_iter, "Soft")
    else:
        if epoch == opt.Q_hard_epoch:
            model.module.save('Q_soft')
            model.module.netE.train()
        for i, data in enumerate(dataset, start=epoch_iter):
            total_steps, epoch_iter = train(data, epoch, total_steps, epoch_iter, "Hard")
    
    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    ### linearly decay learning rate after certain iterations
    #if epoch > opt.niter:
    #    model.module.update_learning_rate()
    if epoch % 100 == 0:
        model.module.update_learning_rate_custom()
    if epoch >= opt.Q_train_epoch and epoch < opt.Q_hard_epoch:
        Temp = (opt.Q_final - opt.Q_init_Temp) / (opt.Q_hard_epoch - opt.Q_train_epoch) * (
                    epoch - opt.Q_train_epoch) + opt.Q_init_Temp
        model.module.update_Temp(Temp)
        print('Temp is %f'%(Temp))
writter.close()
