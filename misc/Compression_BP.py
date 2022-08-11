from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
from util.visualizer import Visualizer
from torch.autograd import Variable
import torch.nn
from sklearn.cluster import KMeans
from models.networks import  quantizer
from models.Bpgan_VGG_Extractor import Bpgan_VGGLoss
from util.nnls import nnls
import ntpath
import os
import imageio
import librosa
import util.util as util
from pytorch_msssim import ssim, ms_ssim
from models.fixed_point_modules import convert_to_8bit

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.quantize_type = 'scalar'
opt.model ="Bpgan_GAN_Q"
how_many_infer = 200
if_quantization = False
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.eval()
visualizer = Visualizer(opt)

device = torch.device("cuda")

Critiretion = torch.nn.MSELoss().to(device)
## ADMM setting
lr = 0.02
BP_iter = 200
alpha = 16
mu = 0.001

output_path = os.path.join("./BP_Results/", opt.name)
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
"""
inverse_matrix = librosa.filters.mel(sr=opt.sampling_ratio,n_fft=opt.n_fft,n_mels=opt.n_mels)
if opt.feature_loss == True:
    VGG_loss = Bpgan_VGGLoss(d=40, sampling_ratio=opt.sampling_ratio, n_fft=opt.n_fft,n_mels=opt.n_mels,path=None)
for i, data in enumerate(dataset):
    if i>how_many_infer:
        break
    with torch.no_grad():
        input = Variable(data['label']).cuda()
        vector = model.netE(input)
        if i == 0:
            vector_dis = vector.detach().cpu().numpy()
        else:
            vector_tem = vector.detach().cpu().numpy()
            vector_dis = np.concatenate((vector_dis, vector_tem), axis=0)
vector_dis = vector_dis.reshape(-1)
kmeans = KMeans(n_clusters=opt.n_cluster,n_jobs=-1).fit(vector_dis.reshape(-1,1))
center = kmeans.cluster_centers_.flatten()
center = torch.Tensor(kmeans.cluster_centers_).cuda()
Quantizer = quantizer(center=center.flatten(),Temp=10)
Quantizer = Quantizer.cuda()

"""
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    input_label, image = model.encode_input(Variable(data['label']), infer=True)
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        #latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector = Compressed_p.clone().detach().requires_grad_()
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    for itera in range(BP_iter):
        generated_img = model.netDecoder.forward(latent_vector)
        if itera==0:
            img_before_update = generated_img.clone().detach()
        #generated_img = generated_img[:,:,:input_label.size(2), :input_label.size(3)]
        optmize_Com.zero_grad()
        if False:#opt.feature_loss == True:
            vgg_loss = VGG_loss(generated_img,input_label)
        else:
            vgg_loss = 0
        mse_loss = Critiretion(generated_img, input_label)
        msssim_loss =  1.-ms_ssim((generated_img+1)/2, (input_label+1)/2, data_range=1)
        Com_loss = vgg_loss + alpha * mse_loss + msssim_loss
        if itera % 10 == 9 or itera == 0:
            print("{:3d} - {:.8f} - {:.8f}".format(itera+1, mse_loss.item(), msssim_loss.item()))
        Com_loss.backward()
        optmize_Com.step()
    with torch.no_grad():
        latent_vector = convert_to_8bit(latent_vector, 2**8, 'uint8', N=8)
        generated_img = model.netDecoder(latent_vector)


    short_path = ntpath.basename(data['path'][0])
    name_ = os.path.splitext(short_path)[0]
    gen_img = util.tensor2im(generated_img[0], imtype=np.uint8)
    gen_img_0 = util.tensor2im(img_before_update[0], imtype=np.uint8)
    org_img = util.tensor2im(input_label[0], imtype=np.uint8)
    print('Saving ', name_)
    imageio.imwrite(os.path.join(output_path, name_ + '_syn.png'), gen_img)
    imageio.imwrite(os.path.join(output_path, name_ + '_syn_before_update.png'), gen_img_0)
    imageio.imwrite(os.path.join(output_path, name_ + '_real.png'), org_img)
