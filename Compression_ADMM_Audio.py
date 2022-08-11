from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
from util.visualizer import Visualizer
from torch.autograd import Variable
import torch.nn
import util.util as util
from sklearn.cluster import KMeans
from models.networks import  quantizer
from models.Bpgan_VGG_Extractor import Bpgan_VGGLoss
from util.nnls import nnls
import ntpath
import os
import imageio
import librosa
from dahuffman import HuffmanCodec
from models.fixed_point_modules import FixedReLU

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.quantize_type = 'scalar'
opt.model ="Bpgan_GAN_Q"
opt.feature_loss = False
how_many_infer = 10
imtype = np.uint16 if opt.image_bit_num == 16 else np.uint8
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.eval()
visualizer = Visualizer(opt)

device = torch.device("cuda") if len(opt.gpu_ids)>0 else torch.device("cpu")

criterion = torch.nn.MSELoss().to(device)
## ADMM setting
lr = 0.00002 if opt.fixed_point else 0.02
ADMM_iter = opt.ADMM_iter
alpha = 16
mu = 0.001


boundary = 10
threshold = 0.99

model = model.to(device)

postfix = "-{}-8bit".format(opt.ADMM_iter) if opt.fixed_point else "-{}".format(opt.ADMM_iter) 
output_path = os.path.join("./ADMM_Results/", opt.name+postfix)
if os.path.exists(output_path) == False:
    os.makedirs(output_path)

Quantizer = model.Q 
recalculate_centers = False
num_center_bits = 8
if recalculate_centers:
    for i, data in enumerate(dataset):
        if i>how_many_infer:
            break
        with torch.no_grad():
            input_ = Variable(data['label']).cuda() if len(opt.gpu_ids)>0 else Variable(data['label'])
            if input_.size()[2] > input_.size()[3]:
                input_ = input_.permute(0, 1, 3, 2)
            vector = model.netE(input_)
            if i == 0:
                vector_dis = vector.detach().cpu().numpy().reshape(-1)
            else:
                vector_tem = vector.detach().cpu().numpy().reshape(-1)
                vector_dis = np.concatenate((vector_dis, vector_tem), axis=0)
    vector_dis = (vector_dis.reshape(-1)*(2**num_center_bits)).astype('int32')/(2**num_center_bits)
    print("kmeans")
    kmeans = KMeans(n_clusters=min(opt.n_cluster, 2**num_center_bits)).fit(vector_dis.reshape(-1,1))
    center = kmeans.cluster_centers_.flatten()
    center = torch.Tensor(kmeans.cluster_centers_)#.cuda()
    if len(opt.gpu_ids)>0:
        center = center.cuda()
    Quantizer = quantizer(center=center.flatten(),Temp=10)


def evaluate():
    for i, data in enumerate(dataset):
        if i > how_many_infer:
            break
        if opt.model == 'Bpgan_GAN':
            generated, latent_vector = model.inference(data['label'])
        elif opt.model =='Bpgan_GAN_Q':
            generated, latent_vector = model.inference(data['label'],Q_type='Hard', store_code_idx=True)
        else:
            raise ValueError("Invalid model type. Got: ",opt.model)




evaluate()
unique, count = np.unique(np.concatenate(model.Q.code_indices), return_counts=True)
freq = {}#dict(zip(unique, count))

j=0
for i in range(opt.n_cluster):
    if i in unique:
        freq[i] = count[j]
        j += 1
    else:
        freq[i] = 1

codec = HuffmanCodec.from_frequencies(freq)
#codec.print_code_table()

#import pandas as pd
#df = pd.DataFrame(columns=['idx', 'MSE', 'MS-SSIM', 'ADMM', 'lv_g_max', 'lv_g_min', 'lv_g_mean'])

avg_bps = 0.
if opt.fixed_point:
    """
    Profiling activations/gradients for fixed point conversion
    """

    # Set layers into 'observe' mode
    for m in model.modules():
        if hasattr(m, 'mode'):
            m.mode = 'observe'
            m.quantize_gradient=True

    if len(opt.gpu_ids)>0:
        Quantizer = Quantizer.cuda()
    print("End finding centers")

    print("Start Observation")
    for i, data in enumerate(dataset):
        if i >= how_many_infer:
            break
        input_label, image = model.encode_input(Variable(data['label']), infer=True)
        with  torch.no_grad():
            Compressed_p = model.netE.forward(input_label)
            latent_vector = Compressed_p.clone().detach().requires_grad_()
            Z = Quantizer(latent_vector, "Hard")
            eta = torch.zeros(latent_vector.shape)
            if len(opt.gpu_ids)>0:
                eta = eta.cuda()

        optmize_Com = torch.optim.SGD([latent_vector], lr=lr, momentum=0.0)
        for j in range(ADMM_iter):
            generated_img = model.netDecoder.forward(latent_vector)
            generated_img = generated_img[:,:,:,:input_label.size(3)]
            gen_img_Q = model.netDecoder(Quantizer(latent_vector, "Hard"))
            gen_img_Q = gen_img_Q[:,:,:,:input_label.size(3)]
            optmize_Com.zero_grad()
            if opt.feature_loss == True:
                vgg_loss = VGG_loss(generated_img,input_label)
            else:
                vgg_loss = 0.
            mse_loss = criterion(generated_img, input_label)
            mse_loss_q = criterion(gen_img_Q, input_label)
            Com_loss = vgg_loss + alpha * mse_loss + mu * 0.5 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                    latent_vector.shape[0]
            if j % 50 == 49 or j==0:
                print('{:4d} MSE: {:0.4f} ADMM Loss: {:0.4f}'.format(j+1, mse_loss_q.data, Com_loss.data)) 
            Com_loss.backward()
            g_max = torch.max(torch.abs(latent_vector.grad))
            g_mean = torch.mean(torch.abs(latent_vector.grad))
            g_min = torch.min(torch.abs(latent_vector.grad))
            #df = df.append({'idx':i, 'MSE':mse_loss.data.item(), 'ADMM': Com_loss.data.item(), 'lv_g_max':g_max.item(), 'lv_g_mean': g_mean.item(), 'lv_g_min': g_min.item()}, ignore_index=True)
            optmize_Com.step()
            with torch.no_grad():
                Z = Quantizer(latent_vector + eta, "Hard")
                eta = eta + latent_vector - Z

        generated_img = model.netDecoder(Quantizer(latent_vector, "Hard"))
        mse_loss = criterion(generated_img[:,:,:,:input_label.size(3)], input_label)
        print(mse_loss.item())
        generated_img = generated_img[:,:,:,:input_label.size(3)]
        encoded = codec.encode(Quantizer.get_indices(latent_vector).detach().cpu().numpy().reshape(-1))
        bps = len(encoded) * 8 / (input_label.size(3)*64/8000)
        #print(bps, len(encoded))
        avg_bps += bps
        for index in range(input_label.shape[0]):
            gen_img = util.tensor2im(generated_img[index], imtype=imtype)
            org_img = util.tensor2im(input_label[index], imtype=imtype)
            
            short_path = ntpath.basename(data['path'][index])
            name_ = os.path.splitext(short_path)[0]
            print('Saving ', name_)
            imageio.imwrite(os.path.join(output_path, name_ + '_syn.png'), gen_img)
            imageio.imwrite(os.path.join(output_path, name_ + '_real.png'), org_img)

    #df.to_csv(os.path.join(output_path, "ADMM_fp.csv"))

    print("End observation")

    # Set layers into 'quant' mode
    for m in model.modules():
        if hasattr(m, 'mode'):
            m.mode = 'quant'


if opt.show_act_quant:
    for m in model.modules():
        if hasattr(m, 'show_act_quant'):
            m.show_act_quant = True


#df = pd.DataFrame(columns=['idx', 'MSE', 'MS-SSIM', 'ADMM', 'lv_g_max', 'lv_g_min', 'lv_g_mean'])
for i, data in enumerate(dataset):
    input_label, image = model.encode_input(Variable(data['label']), infer=True)
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        latent_vector = Compressed_p.clone().detach().requires_grad_()
        Z = Quantizer(latent_vector, "Hard")
        eta = torch.zeros(latent_vector.shape)
        if len(opt.gpu_ids)>0:
            eta = eta.cuda()

    best_lv = latent_vector.detach().clone()
    min_loss = 99999.

    if opt.fixed_point:
        optmize_Com = torch.optim.SGD([latent_vector], lr=lr, momentum=0.0)
    else:
        optmize_Com = torch.optim.Adam([latent_vector], lr=lr)

    for j in range(ADMM_iter):
        generated_img = model.netDecoder.forward(latent_vector)
        generated_img = generated_img[:,:,:,:input_label.size(3)]
        gen_img_Q = model.netDecoder(Quantizer(latent_vector, "Hard"))
        gen_img_Q = gen_img_Q[:,:,:,:input_label.size(3)]
        optmize_Com.zero_grad()
        if opt.feature_loss:
            vgg_loss = VGG_loss(generated_img,input_label)
        else:
            vgg_loss = 0.
        mse_loss = criterion(generated_img, input_label)
        mse_loss_q = criterion(gen_img_Q, input_label)
        Com_loss = vgg_loss + alpha * mse_loss + mu * 0.5 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                   latent_vector.shape[0]
        if min_loss > Com_loss.item():
            min_loss = Com_loss.item()
            best_lv = latent_vector.detach().clone()
        if True:#j % 10 == 9 or j==0:
            print('{:4d} MSE Hard: {:0.4f} MSE NQ: {:0.4f} ADMM Loss: {:0.4f}'.format(j+1, mse_loss_q.data, mse_loss.data, Com_loss.data)) 
        Com_loss.backward()
        g_max = torch.max(torch.abs(latent_vector.grad))
        g_mean = torch.mean(torch.abs(latent_vector.grad))
        g_min = torch.min(torch.abs(latent_vector.grad))
        #df = df.append({'idx':i, 'MSE':mse_loss.data.item(), 'ADMM': Com_loss.data.item(), 'lv_g_max':g_max.item(), 'lv_g_mean': g_mean.item(), 'lv_g_min': g_min.item()}, ignore_index=True)
        optmize_Com.step()
        with torch.no_grad():
            Z = Quantizer(latent_vector + eta, "Hard")
            eta = eta + latent_vector - Z

    generated_img = model.netDecoder(Quantizer(best_lv, "Hard"))
    mse_loss = criterion(generated_img[:,:,:,:input_label.size(3)], input_label)
    print(mse_loss.item())
    generated_img = generated_img[:,:,:,:input_label.size(3)]
    encoded = codec.encode(Quantizer.get_indices(latent_vector).detach().cpu().numpy().reshape(-1))
    bps = len(encoded) * 8 / (input_label.size(3)*64/8000)
    avg_bps += bps
    for index in range(input_label.shape[0]):
        gen_img = util.tensor2im(generated_img[index], imtype=imtype)
        org_img = util.tensor2im(input_label[index], imtype=imtype)
        
        short_path = ntpath.basename(data['path'][index])
        name_ = os.path.splitext(short_path)[0]
        print(i+1, 'Saving ', name_)
        postfix = '_syn-8bit.png' if opt.fixed_point else '_syn.png'
        imageio.imwrite(os.path.join(output_path, name_ + postfix), gen_img)

    #df.to_csv(os.path.join(output_path, "ADMM_8bit.csv"))
