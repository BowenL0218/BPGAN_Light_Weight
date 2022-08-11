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
from pytorch_msssim import ssim, ms_ssim
from dahuffman import HuffmanCodec

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
#opt.quantize_type = 'scalar'
opt.model ="Bpgan_GAN_Q"
opt.feature_loss = False
how_many_infer = 10
imtype = np.uint16 if opt.image_bit_num == 16 else np.uint8
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.eval()
visualizer = Visualizer(opt)

#device = torch.device("cuda")
device = torch.device("cuda") if len(opt.gpu_ids)>0 else torch.device("cpu")

criterion_mse = torch.nn.MSELoss()#L1Loss()
criterion = torch.nn.CrossEntropyLoss()
## ADMM setting
lr = 0.02
ADMM_iter = opt.ADMM_iter
BP_iter = opt.ADMM_iter
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
                vector_dis = vector.detach().cpu().numpy()
            else:
                vector_tem = vector.detach().cpu().numpy()
                print(vector_tem.shape, vector_dis.shape)
                vector_dis = np.concatenate((vector_dis, vector_tem), axis=0)
    vector_dis = vector_dis.reshape(-1, 4) if opt.quantize_type == 'vector' else vector_dis.reshape(-1)
    vector_dis = (vector_dis*(2**num_center_bits)).astype('int32')/(2**num_center_bits)
    print("kmeans")
    if opt.quantize_type == 'scalar':
        kmeans = KMeans(n_clusters=min(opt.n_cluster, 2**num_center_bits)).fit(vector_dis.reshape(-1,1))
        center = kmeans.cluster_centers_.flatten()
    elif opt.quantize_type == 'vector':
        kmeans = KMeans(n_clusters=opt.n_cluster).fit(vector_dis)
        center = kmeans.cluster_centers_
    center = torch.Tensor(center)
    if len(opt.gpu_ids)>0:
        center = center.cuda()
    model.update_center(center)
    Quantizer = model.Q

def evaluate():
    for i, data in enumerate(dataset):
        if opt.model == 'Bpgan_GAN':
            generated, latent_vector = model.inference(data['label'])
        elif opt.model =='Bpgan_GAN_Q':
            generated, latent_vector = model.inference(data['label'],Q_type='Hard', store_code_idx=True)
        else:
            raise ValueError("Invalid model type. Got: ",opt.model)

evaluate()
unique, count = np.unique(np.concatenate(model.Q.code_indices), return_counts=True)
print(unique)
print(count)
freq = {}
j = 0
for i in range(opt.n_cluster):
    if i in unique:
        freq[i] = count[j]
        j += 1
    else:
        freq[i] = 1

print(freq)

#freq = dict(zip(unique, count))
codec = HuffmanCodec.from_frequencies(freq)
codec.print_code_table()


import pandas as pd
df = pd.DataFrame(columns=['idx', 'MSE', 'MS-SSIM', 'ADMM', 'lv_g_max', 'lv_g_min', 'lv_g_mean'])
if len(opt.gpu_ids)>0:
    Quantizer = Quantizer.cuda()
print("End finding centers")






avg_bpp = 0.
for i, data in enumerate(dataset):
    input_label, image = model.encode_input(Variable(data['label']), infer=True)
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        #Compressed_p = Quantizer(Compressed_p, "Hard")
        latent_vector = Compressed_p.clone().detach().requires_grad_()
        target_vector = Compressed_p.clone().detach()
        target_labels = Quantizer.get_indices(target_vector)

        encoded = codec.encode(Quantizer.get_indices(target_vector).detach().cpu().numpy().reshape(-1))
        bpp = len(encoded)*8/(input_label.size(2)*input_label.size(3))
        print('bpp',bpp, len(encoded))
        avg_bpp += bpp

    optimize_Com = torch.optim.Adam([latent_vector], lr=lr)
    for j in range(BP_iter):
        optimize_Com.zero_grad()
        generated_img = model.netDecoder.forward(latent_vector)
        encoded = model.netE(generated_img)
        encoded = encoded.view(encoded.size()+(1,))

        diffs = torch.abs(encoded - Quantizer.center.view(1,1,1,1,-1))
        bp_loss = criterion(-diffs.view(-1, opt.n_cluster), target_labels.view(-1))

        if j % 50 == 49 or j==0:
            with torch.no_grad():
                generated_image = model.netDecoder(latent_vector)

                ms_ssim_val = ms_ssim((generated_img+1)/2, (input_label+1)/2, data_range=1)
                mse_image = criterion_mse(generated_img, input_label)

                print('BP   - {:4d} CE Loss: {:e} MSE Image: {:0.4f} MS-SSIM Image: {:0.4f}'.format(j+1, bp_loss.item(), mse_image.item(), ms_ssim_val.item())) 
        bp_loss.backward()
        optimize_Com.step()

    with torch.no_grad():
        generated_img = model.netDecoder(latent_vector)
        generated_img = generated_img[:,:,:input_label.size(2),:input_label.size(3)]
        mse_loss = criterion_mse(generated_img, input_label)
        print('final mse', mse_loss.item())

        for index in range(input_label.shape[0]):
            gen_img = util.tensor2im(generated_img[index])
            org_img = util.tensor2im(input_label[index])
            
            short_path = ntpath.basename(data['path'][index])
            name_ = os.path.splitext(short_path)[0]
            print('Saving ', name_)
            if opt.fixed_point:
                imageio.imwrite(os.path.join(output_path, name_ + '_syn_8bit.png'), gen_img)
            else:
                imageio.imwrite(os.path.join(output_path, name_ + '_syn.png'), gen_img)
            imageio.imwrite(os.path.join(output_path, name_ + '_real.png'), org_img)

print("Avg bpp", avg_bpp/(i))
df.to_csv("ADMM.csv")
