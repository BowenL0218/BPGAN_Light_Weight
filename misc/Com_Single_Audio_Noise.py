from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.nnls import nnls
import ntpath
import os
import imageio
import librosa
from torch.autograd import Variable
import numpy as np
import torch
from models.networks import  quantizer
import scipy.io as sio
from Data_Processing.numba_pghi import audio_spliting, audio_recovery
opt = TestOptions().parse(save=False)
model = create_model(opt)
inverse_matrix = librosa.filters.mel(sr=opt.sampling_ratio,n_fft=opt.n_fft,n_mels=opt.n_mels)
output_path = opt.output_path
input_audio_path = opt.input_file
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
center = np.load(os.path.join(opt.checkpoints_dir,opt.name,"Quantization_Center.npy"))
center = torch.Tensor(center).cuda()
Quantizer = quantizer(center=center.flatten(),Temp=10)
Quantizer = Quantizer.cuda()
if opt.sampling_ratio == 8000 and opt.n_fft == 256 and opt.n_mels == 64:
    sampling_ratio = 8000
    fft_hop_size = 64
    fft_window_length = 256
    imagesize = 64
    L = 64*64
elif opt.sampling_ratio == 16000 and opt.n_fft == 512 and opt.n_mels == 128:
    sampling_ratio = 16000
    fft_hop_size = 128
    fft_window_length = 512
    imagesize = 128
    L = 128*128
else:
    raise NotImplementedError("Not implemented settings")
spliting_audio, ratio = audio_spliting(input_audio_path,fft_hop_size=fft_hop_size,fft_window_length=fft_window_length,clipBelow=-10,sampling_fre=sampling_ratio,imagesize=imagesize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i in range(len(spliting_audio)):
    input_label = torch.Tensor(spliting_audio[i]).to(device).unsqueeze(0).unsqueeze(0)
    latent_vector = Quantizer(model.netE(input_label),"Hard")
    generated = model.netDecoder(latent_vector)
    gen_img = generated[0].detach().cpu().numpy()
    org_img = input_label[0].detach().cpu().numpy()
    gen_img = gen_img * 0.5 - 0.5
    org_img = org_img * 0.5 - 0.5
    gen_img = np.exp(10 * (gen_img))
    org_img = np.exp(10 * (org_img))
    inverse_gen = np.abs(nnls(inverse_matrix, gen_img[0, :, :]))
    inverse_org = np.abs(nnls(inverse_matrix, org_img[0, :, :]))
    inverse_gen_img = (inverse_gen / np.max(inverse_gen.flatten()) * 65535).astype(np.uint16)
    inverse_org_img = (inverse_org / np.max(inverse_org.flatten()) * 65535).astype(np.uint16)
    short_path = ntpath.basename(input_audio_path)
    name_ = os.path.splitext(short_path)[0]
    imageio.imwrite(os.path.join(output_path, name_ +str(i)+ '_syn.png'), inverse_gen_img)
    imageio.imwrite(os.path.join(output_path, name_ +str(i)+ '_real.png'), inverse_org_img)
    sio.savemat(os.path.join(output_path, name_ +str(i)+ 'vector.mat'),{'vector':latent_vector.detach().cpu().numpy()})



