from collections import OrderedDict
from options.test_options import TestOptions
from options.train_Q_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
from util.visualizer import Visualizer
from torch.autograd import Variable
import torch.nn
import util.util as util
from sklearn.cluster import KMeans
from models.networks import  quantizer, VGGFeatureExtractor
from models.Bpgan_VGG_Extractor import Bpgan_VGGLoss
from util.nnls import nnls
import ntpath
import os
import imageio
import librosa
from pytorch_msssim import ssim, ms_ssim
from models.fixed_point_modules import MinMaxObserver, convert_to_8bit
from models.fixed_point_modules import FixedReLU, FixedAdd, FixedInputQuantizer, FixedTanh, FixedSigmoid, FixedConv2d, FixedConvTranspose2d
from models.networks import ConvBNReLU, ConvTransposeBNReLU
from dahuffman import HuffmanCodec
import torch.nn.utils.prune as prune

test_mode = False
num_bias_centers = 128
if test_mode:
    first_pretrain_epoch = 0
    second_pretrain_epoch = 0
    how_many_infer = 100
    ADMM_iter = 4
    ADMM_epoch = 10
else:
    first_pretrain_epoch = 0
    second_pretrain_epoch = 0
    how_many_infer = 500
    ADMM_iter = 40
    ADMM_epoch = 10


num_weight_centers_encoder = 16
num_weight_centers_decoder = 8
encoder_sparsity = 0.4
decoder_sparsity = 0.8

opt = TestOptions().parse(save=False)
opt.nThreads = 2  # test code only supports nThreads = 1
opt.batchSize = 64  # test code only supports batchSize = 1
opt.serial_batches = False  # no shuffle
opt.no_flip = False  # no flip
opt.quantize_type = 'scalar'
opt.model ="Bpgan_GAN_Q"
opt.feature_loss = False
opt.qint8 = True
opt.phase = 'train'
data_loader = CreateDataLoader(opt)

opt.no_flip = True
opt.nThreads = 1
opt.batchSize=1
opt.serial_batches = True
opt.phase = 'test'
test_data_loader = CreateDataLoader(opt)
test_dataset = test_data_loader.load_data()
for i, data in enumerate(test_dataset):
    if i==10:
        test_data = data
        break
opt.no_flip = False
opt.nThreads = 2
opt.batchSize= 64
opt.serial_batches = False
opt.phase = 'train'

dataset = data_loader.load_data()
model = create_model(opt)
model.eval()
visualizer = Visualizer(opt)

test_image, _ = model.encode_input(Variable(test_data['label']), infer=True)
#device = torch.device("cuda")
device = torch.device("cuda") if len(opt.gpu_ids)>0 else torch.device("cpu")

criterion = torch.nn.MSELoss().to(device)
## ADMM setting
lr = 0.0002
alpha = 10
mu = 0.01

threshold = 0.99

model = model.to(device)
if not opt.fuse_layers:
    for m in model.modules():
        if isinstance(m, ConvBNReLU) or isinstance(m, ConvTransposeBNReLU):
            m.fuse_layers()
Quantizer = model.Q

output_path = os.path.join("./ADMM_Net_Comp/", opt.name)
if os.path.exists(output_path) == False:
    os.makedirs(output_path)

generated_img, _ = model.inference(test_image, "Hard")
gen_img = util.tensor2im(generated_img[0])
imageio.imwrite(os.path.join(output_path, 'orig.png'), gen_img)

total_params = list(model.netE.parameters()) + list(model.netDecoder.parameters())


def evaluate():
    model.eval()
    for i, data in enumerate(dataset):
        if i > 0:#how_many_infer:
            break
        if opt.model == 'Bpgan_GAN':
            generated, latent_vector = model.inference(data['label'])
        elif opt.model =='Bpgan_GAN_Q':
            generated, latent_vector = model.inference(data['label'],Q_type='Hard', store_code_idx=True)
        else:
            raise ValueError("Invalid model type. Got: ",opt.model)




for m in model.modules():
    if True:#isinstance(m, FixedReLU) or isinstance(m, FixedInputQuantizer) or isinstance(m, FixedTanh) or isinstance(m, FixedSigmoid):
        m.mode = 'observe'
        m.quantize_gradient=False
evaluate()


for m in model.modules():
    if isinstance(m, FixedReLU) or isinstance(m, FixedInputQuantizer) or isinstance(m, FixedSigmoid):# or isinstance(m, FixedTanh)
        pass#m.mode = 'quant'

def nonlinear_quantize(w_, c_):
    idx = torch.argmin(torch.abs(w_.data.view(-1, 1) - c_), dim=1)
    return c_[:,idx].view(w_.size())

def get_fractional(x):
    max_val = np.max((np.abs(x.max()), np.abs(x.min())))
    if max_val > 127.:
        n_fractional = 0
    elif max_val < 1.:
        n_fractional = 7
    else:
        n_fractional = 7
        while int(max_val)//(2**(7-n_fractional))==0:
            n_fractional -= 1
    return n_fractional


for p in total_params:
    with torch.no_grad():
        p.data = p.clamp_(-0.5, 0.5)

optmize_Com = torch.optim.Adam(total_params, lr=lr, weight_decay=0e-3)
for e in range(first_pretrain_epoch):
    for k, data in enumerate(dataset):
        if k > 400:
            break
        input_label, image = model.encode_input(Variable(data['label']), Variable(data['image']), infer=True)

        msssim_loss = 1
        optmize_Com.zero_grad()
        generated_img, _ = model.inference(input_label, "None")
        mse_loss = criterion(generated_img, input_label)
        if k % 200 == 0:#j % 50 == 49 or j==0:
            print('{:4d} MSE: {:0.4f} '.format(k+1, mse_loss.item()))
        mse_loss.backward()
        optmize_Com.step()





modules = []
full_modules = []
module_idx = 0
for name, module in model.named_modules():
    if isinstance(module, FixedConvTranspose2d) or isinstance(module, FixedConv2d):
        full_modules.append(module)
        #if module_idx >= 7:
        if module_idx >= 7 :
            #prune.l1_unstructured(module, name='weight', amount=decoder_sparsity)
            #prune.l1_unstructured(module, name='bias', amount=0.0)
            module.num_weight_centers = num_weight_centers_decoder
            module.sparsity = decoder_sparsity
            #module.num_weight_centers = num_weight_centers_fc
        else:
            #prune.l1_unstructured(module, name='weight', amount=encoder_sparsity)
            #prune.l1_unstructured(module, name='bias', amount=0.0)
            module.num_weight_centers = num_weight_centers_encoder
            module.sparsity = encoder_sparsity
            #module.num_weight_centers = num_weight_centers_conv
            #prune.ln_structured(module, 'weight', amount=0.3, n=2, dim=0)
            #prune.ln_structured(module, 'weight', amount=0.3, n=2, dim=1)
            #modules.append((module, 'weight'))

        module_idx += 1

model.train()

def project_sparse_weight(weight, sparsity):
    assert sparsity <= 1. and sparsity >= 0.

    with torch.no_grad():
        location = int(np.floor(float(weight.nelement()) * sparsity))
        sorted, indices = torch.sort(torch.abs(weight.view(-1)), descending=False)
        val = sorted[location]
        w = weight.clone().detach()
        w[torch.abs(w) < val] = 0.
    return w

def get_mask(weight, sparsity):
    assert sparsity <= 1. and sparsity >= 0.

    with torch.no_grad():
        location = int(np.floor(float(weight.nelement()) * sparsity))
        sorted, indices = torch.sort(torch.abs(weight.view(-1)), descending=False)
        val = sorted[location]
        w = weight.clone().detach()
        w[torch.abs(w) < val] = 0.
        w[torch.abs(w) >= val] = 1.
    return w





optmize_Com = torch.optim.Adam(total_params, lr=lr, weight_decay=0e-2)
for e in range(second_pretrain_epoch):
    for k, data in enumerate(dataset):
        if k > 30:
            break
        input_label, image = model.encode_input(Variable(data['label']), Variable(data['image']), infer=True)

        for module in full_modules:
            with torch.no_grad():
                module.Z = project_sparse_weight(module.weight, module.sparsity)
                module.eta = torch.zeros_like(module.weight).cuda()

        for j in range(20):
            optmize_Com.zero_grad()
            generated_img, _ = model.inference(input_label, "None")
            mse_loss = criterion(generated_img, input_label)
            Com_loss =  alpha * mse_loss +  mu / 2.0 * torch.mean(torch.stack([torch.norm(module.weight - module.Z + module.eta, 2) ** 2  for module in full_modules]))
            if k % 10 == 0 and j % 10 == 0:#j % 50 == 49 or j==0:
                print('{:3d} {:3d} MSE: {:0.4f} ADMM: {:0.4f}'.format(k+1, j+1, mse_loss.item(), Com_loss.item()))
            Com_loss.backward()
            optmize_Com.step()
            with torch.no_grad():
                for module in full_modules:
                    with torch.no_grad():
                        module.Z = project_sparse_weight(module.weight + module.eta, module.sparsity)
                        module.eta += module.weight.data - module.Z



        for module in full_modules:
            with torch.no_grad(): 
                module.weight.data = project_sparse_weight(module.weight, module.sparsity)

for module in full_modules:
    with torch.no_grad():
        mask = get_mask(module.weight, module.sparsity)
        prune.custom_from_mask(module, 'weight', mask)



for name, module in model.named_modules():
    if isinstance(module, FixedConvTranspose2d) or isinstance(module, FixedConv2d):
        print(name, torch.sum(module.weight == 0).item()/float(module.weight.nelement()))


save_dir = os.path.join(opt.checkpoints_dir, opt.name)
if os.path.exists(os.path.join(save_dir, 'pruned_{:.2f}_{:.2f}_net_E.pth'.format(encoder_sparsity, decoder_sparsity))):
    model.load_network(model.netE, 'E', 'pruned_{:.2f}_{:.2f}'.format(encoder_sparsity, decoder_sparsity), save_dir)
    model.load_network(model.netDecoder, 'Decoder', 'pruned_{:.2f}_{:.2f}'.format(encoder_sparsity, decoder_sparsity), save_dir)

    if False:#test_mode:
        module_idx = 0
        from collections import OrderedDict
        for name, module in model.named_modules():
            if isinstance(module, FixedConvTranspose2d) or isinstance(module, FixedConv2d):
                weight_orig_data = module.weight_orig.detach().clone()
                bias_orig_data = module.bias_orig.detach().clone()

                prune.remove(module, 'weight')
                prune.remove(module, 'bias')

                with torch.no_grad():
                    module.weight.data = weight_orig_data.data
                    module.bias.data = bias_orig_data.data

                full_modules.append(module)
                if module_idx >= 6 :
                    prune.l1_unstructured(module, name='weight', amount=decoder_sparsity)
                    prune.l1_unstructured(module, name='bias', amount=0.0)
                    module.num_weight_centers = num_weight_centers_decoder
                else:
                    prune.l1_unstructured(module, name='weight', amount=encoder_sparsity)
                    prune.l1_unstructured(module, name='bias', amount=0.0)
                    module.num_weight_centers = num_weight_centers_encoder

                module_idx += 1



elif not test_mode:
    model.save('pruned_{:.2f}_{:.2f}'.format(encoder_sparsity, decoder_sparsity), save_netD=False)


with open("log.txt", "w+") as f:
    f.write("")


model.to(device)
centers_list = []
skip_list = []
module_list = []
for i, weight in list(enumerate(total_params)):

    num_weight_centers = full_modules[i//2].num_weight_centers if len(weight.size())!=1 else num_bias_centers

    if len(weight.view(-1)) > num_weight_centers:
        with torch.no_grad():
            w = weight.data.detach().clone().cpu().numpy().reshape(-1, 1)
            space = torch.linspace(weight.min(), weight.max(), steps=num_weight_centers)
            kmeans = KMeans(n_clusters=min(num_weight_centers, len(np.unique(w))), init=space.reshape(-1, 1), n_init=1)
            centers = kmeans.fit(w).cluster_centers_.reshape(-1)
                    
            n_fractional = 7#get_fractional(centers) 
            centers = (centers * (2**n_fractional)).astype("int32").astype("float32") #/ (2**n_fractional)
            centers = np.clip(centers, -2**n_fractional+1, 2**n_fractional-1)
            centers = centers / (2**n_fractional)
            centers = centers.reshape(1, -1)
            centers = np.unique(centers, axis=1)
            print('Num Centers:' ,len(centers.reshape(-1)), centers.max(), centers.min())
            centers = torch.Tensor(centers).cuda()
            centers_list.append(centers)
            module_list.append(full_modules[i//2])
    else:
        skip_list.append(i)

                
#optmize_Com = torch.optim.SGD(total_params, lr=lr, momentum=0.9)
optmize_Com = torch.optim.Adam(total_params, lr=lr)
for e in range(ADMM_epoch):
    for k, data in enumerate(dataset):
        weight_list = []
        weight_clone_list = []
        Z_list = []
        eta_list = []
        if k >= how_many_infer:
            break
        input_label, image = model.encode_input(Variable(data['label']), Variable(data['image']), infer=True)

        count = 0
        for i, weight in list(enumerate(total_params)):
            if i in skip_list:
                continue
            weight_list.append(weight)
            weight_clone = weight.clone().detach().requires_grad_()
            weight_clone_list.append(weight_clone)
            weight = weight.cuda()

            with torch.no_grad():
                Z_list.append(nonlinear_quantize(weight_clone.data, centers_list[count]))
                eta = torch.zeros(weight_clone.shape)
                if len(opt.gpu_ids)>0:
                    eta = eta.cuda()
                eta_list.append(eta)
            count += 1

        for j in range(ADMM_iter):
            optmize_Com.zero_grad()
            generated_img, _ = model.inference(input_label, "None")
            mse_loss = criterion(generated_img, input_label)
            Com_loss =  alpha * mse_loss +  mu / 2.0 * torch.mean(torch.stack([torch.norm(weight - Z + eta, 2) ** 2  for weight, Z, eta in zip(weight_list, Z_list, eta_list)]))
            if k % 10 == 0 and j % 10 == 0:
                print('{:3d} {:3d} MSE: {:0.4f} ADMM Loss: {:0.4f}'.format(k+1, j+1, mse_loss.item(), Com_loss.item())) 
                    
            Com_loss.backward()
            optmize_Com.step()
            with torch.no_grad():
                for weight, weight_clone, Z, eta, centers in zip(weight_list, weight_clone_list, Z_list, eta_list, centers_list):
                    weight_clone.data = weight.data
                    Z = nonlinear_quantize(weight_clone.data + eta, centers)
                    eta = eta + weight.data - Z
        with torch.no_grad():
            for i, (weight, weight_clone, centers) in enumerate(zip(weight_list, weight_clone_list, centers_list)):
                if len(weight.size()) > 1:
                    weight.data = module_list[i].weight_mask * nonlinear_quantize(weight_clone.data, centers)
                else:
                    weight.data = nonlinear_quantize(weight_clone.data, centers)

    with torch.no_grad():
        generated_img, _ = model.inference(test_image, "Hard")
        gen_img = util.tensor2im(generated_img[0])
        mse_loss = criterion(generated_img[:,:,:test_image.size(2),:test_image.size(3)], test_image)
        print("Test MSE:", mse_loss.item())
        with open("log.txt", "a+") as f:
            f.write("Test MSE: {}\n".format(mse_loss.item()))


for module in full_modules:
    prune.remove(module, 'weight')




model.save('ADMM_Q_pruned_full', save_netD=False)
