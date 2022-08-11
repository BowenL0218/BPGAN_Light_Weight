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


model.module.netDecoder.eval()
optimizer = torch.optim.Adam(model.module.netE.parameters(), lr=opt.lr)

criterion_mse = nn.MSELoss()
criterion_xe = nn.CrossEntropyLoss()

def train(data, epoch, total_steps, epoch_iter, Q_type):
    
    iter_start_time = time.time()
    total_steps += opt.batchSize
    epoch_iter += opt.batchSize

    # whether to collect output images
    save_fake = total_steps % opt.display_freq == 0

    ############## Forward Pass ######################
    input_label, _ = model.module.encode_input(data['label'].cuda(), infer=True)
    with torch.no_grad():
        initial_z = model.module.netE(input_label)
        latent_vector = initial_z.clone().detach().requires_grad_()

        Z = model.module.Q(latent_vector, "Hard")
        eta = torch.zeros(latent_vector.shape)
        if len(opt.gpu_ids)>0:
            eta = eta.cuda()

    
    optimizer_bp = torch.optim.Adam([latent_vector], lr=0.05)
    for j in range(50):
        optimizer_bp.zero_grad()
        reconst = model.module.netDecoder(latent_vector)
        mse_loss = criterion_mse(reconst, input_label)
        com_loss = mse_loss  + 0.001 * 0.5 * torch.norm(latent_vector - Z + eta, 2) ** 2 / latent_vector.shape[0]
        com_loss.backward()
        optimizer_bp.step()

        if j==0:
            print("ADMM, 0: ", mse_loss.item())
    print("ADMM, {}: {}".format(j, mse_loss.item()))

    with torch.no_grad():
        target_labels = model.module.Q.get_indices(latent_vector)
    
    optimizer.zero_grad()
    encoded = model.module.netE(input_label)
    encoded = encoded.view(encoded.size()+(1,))
    diffs = torch.abs(encoded - model.module.Q.center.view(1,1,1,1,-1))
    xe_loss = criterion_xe(-diffs.view(-1, opt.n_cluster), target_labels.view(-1))
    xe_loss.backward()
    optimizer.step()

    with torch.no_grad():
        fake_image = model.module.forward(input_label, Q_type="Hard")
        mse_loss = criterion_mse(fake_image, input_label)

    loss_dict = {'Cross Entropy': xe_loss,
                 'MSE': mse_loss
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
