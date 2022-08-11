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
from torch.quantization import QConfig, MinMaxObserver
import copy
from dahuffman import HuffmanCodec

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.quantize_type = 'scalar'
opt.model ="Bpgan_GAN_Q"
opt.feature_loss = False
how_many_infer = 20
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.eval()
#netE = torch.jit.load("8bit/qmodel_E.pth")
#netDecoder = torch.jit.load("8bit/qmodel_D.pth")

netDecoder = copy.deepcopy(model.netDecoder)

model.netE.fuse_model()
model.netDecoder.fuse_model()
netDecoder.fuse_model()

act_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=act_observer,
                  weight=weight_observer)
model.netE.qconfig = qconfig
model.netDecoder.qconfig=qconfig
torch.quantization.prepare(model.netE, inplace=True)
torch.quantization.prepare(model.netDecoder, inplace=True)

recalculate_centers = False
if recalculate_centers:
    for i, data in enumerate(dataset):
        if i == 0:
            continue
        if i>how_many_infer:
            break
        with torch.no_grad():
            input_ = Variable(data['label'])
            if input_.size()[2] > input_.size()[3]:
                input_ = input_.permute(0, 1, 3, 2)
            vector = model.netE(input_)
            if opt.quantize_type == 'vector':
                vector = vector.view(-1,4)
            if i == 1:
                vector_dis = vector.detach().cpu().numpy()
            else:
                vector_tem = vector.detach().cpu().numpy()
                vector_dis = np.concatenate((vector_dis, vector_tem), axis=0)
    
    print('kmeans')
    if opt.quantize_type == 'scalar':
        vector_dis = vector_dis.reshape(-1)
        kmeans = KMeans(n_clusters=opt.n_cluster).fit(vector_dis.reshape(-1,1))
        center = kmeans.cluster_centers_.flatten()
    elif opt.quantize_type == 'vector':
        vector_dis = vector_dis.reshape(-1,4)
        kmeans = KMeans(n_clusters=opt.n_cluster).fit(vector_dis)
        center = kmeans.cluster_centers_
    '''
    vector_dis = vector_dis.reshape(-1)
    print("kmeans")
    kmeans = KMeans(n_clusters=opt.n_cluster).fit(vector_dis.reshape(-1,1))
    center = kmeans.cluster_centers_.flatten()
    '''
    center = torch.Tensor(kmeans.cluster_centers_)#.cuda()
    if len(opt.gpu_ids)>0:
        center = center.cuda()
    model.Q = quantizer(center=center.flatten(),Temp=10)




centers = model.Q.center.data.detach().cpu()
fx_centers = []
num_center_bits = 8
for c in centers:
    c = int(c*(2**num_center_bits))/(2**num_center_bits)
    if c not in fx_centers:
        fx_centers.append(c)
fx_centers = np.array(fx_centers).astype('float32') 
model.Q.update_center(torch.from_numpy(fx_centers).float())
print(len(fx_centers))


def evaluate():
    for i, data in enumerate(dataset):
        if opt.model == 'Bpgan_GAN':
            generated, latent_vector = model.inference(data['label'])
        elif opt.model =='Bpgan_GAN_Q':
            generated, latent_vector = model.inference(data['label'],Q_type='Hard', store_code_vals=True)
        else:
            raise ValueError("Invalid model type. Got: ",opt.model)

evaluate()
unique, count = np.unique(np.concatenate(model.Q.code_vals+[fx_centers]), return_counts=True)
freq = dict(zip(unique, count))
print(freq)
codec = HuffmanCodec.from_frequencies(freq)
codec.print_code_table()


netEQ = torch.quantization.convert(model.netE)
netDecoderQ = torch.quantization.convert(model.netDecoder)
print('Model quantized')
model.netE = netEQ
model.netDecoder = netDecoderQ

visualizer = Visualizer(opt)
device = torch.device("cuda") if len(opt.gpu_ids)>0 else torch.device("cpu")

criterion = torch.nn.MSELoss().to(device)
## ADMM setting
lr = 0.01
ADMM_iter = 400
alpha = 16
mu = 0.001


threshold = 0.99

model = model.to(device)

output_path = os.path.join("./ADMM_Results_8bit/", opt.name)
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
# A = librosa.filters.mel(sr=opt.sampling_ratio,n_fft=opt.n_fft,n_mels=40)
# B = librosa.filters.mel(sr=opt.sampling_ratio,n_fft=512,n_mels=128)
# C = A.dot(np.linalg.pinv(B))
# Transform_tensor = torch.Tensor(C).cuda()


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model.netE)
print_size_of_model(model.netDecoder)

if len(opt.gpu_ids)>0:
    model.Q = model.Q.cuda()
print("End finding centers")

import pandas as pd
df = pd.DataFrame(columns=['idx', 'MSE', 'MS-SSIM', 'ADMM'])
avg_bpp = 0.
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    input_label, image = model.encode_input(Variable(data['label']), infer=True)
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        latent_vector = Compressed_p.clone().detach().requires_grad_()
        Z = model.Q(latent_vector, "Hard")
        eta = torch.zeros(latent_vector.shape)
        if len(opt.gpu_ids)>0:
            eta = eta.cuda()
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    #optmize_Com = torch.optim.SGD([latent_vector], lr=1.)
    msssim_loss = 1
    j = 0
    #for itera in range(ADMM_iter):
    while msssim_loss > 1 - threshold and j < ADMM_iter:
        optmize_Com.zero_grad()
        generated_img = netDecoder.forward(latent_vector)
        with torch.no_grad():
            gen_img_Q = model.netDecoder.forward(latent_vector)
        mse_loss = criterion(generated_img, input_label)
        mse_loss_8bit = criterion(gen_img_Q, input_label)
        msssim_loss = 1 - ms_ssim((gen_img_Q+1)/2, (input_label+1)/2, data_range=1)
        Com_loss = alpha * (mse_loss ) + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                   latent_vector.shape[0]

        Com_loss_Q = alpha * (mse_loss_8bit ) + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                   latent_vector.shape[0]
        summary = '{:4d} MSE: {:0.4f} MSE 8bit: {:0.4f} MS-SSIM Loss: {:0.4f} ADMM Loss: {:0.4f}'.format(j+1, mse_loss.data, mse_loss_8bit.data, msssim_loss.data, Com_loss_Q.data)
        if j % 100 == 99:
            print(summary) 
        df = df.append({'idx':i, 'MSE':mse_loss.data.item(), 'MS-SSIM': msssim_loss.data.item(), 'ADMM': Com_loss.data.item()}, ignore_index=True)
        Com_loss.backward(Com_loss_Q)
        #latent_vector.grad += 0.001*torch.randn(latent_vector.size())
        optmize_Com.step()
        with torch.no_grad():
            Z = model.Q(latent_vector + eta, "Hard")
            eta = eta + latent_vector - Z
        j = j + 1

    encoded = codec.encode(model.Q(latent_vector, "Hard").detach().cpu().numpy().reshape(-1))
    bpp = len(encoded)*8/(input_label.size(2)*input_label.size(3))
    print(bpp)
    avg_bpp += bpp
    generated_img = model.netDecoder(model.Q(latent_vector, "Hard"))
    for index in range(input_label.shape[0]):
        gen_img = util.tensor2im(generated_img[index])
        org_img = util.tensor2im(input_label[index])
        
        short_path = ntpath.basename(data['path'][index])
        name_ = os.path.splitext(short_path)[0]
        print('Saving ', name_)
        imageio.imwrite(os.path.join(output_path, name_ + '_syn.png'), gen_img)
        imageio.imwrite(os.path.join(output_path, name_ + '_real.png'), org_img)

print("Average bpp: ", avg_bpp/i)