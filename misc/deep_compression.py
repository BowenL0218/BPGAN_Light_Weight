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


"""
Compression Settings
"""
test_mode = False
lr = 0.0001
if test_mode:
    first_pretrain_epoch = 1
    second_pretrain_epoch = 1
    quant_epoch = 1
    bp_iter = 200
else:
    first_pretrain_epoch = 0
    second_pretrain_epoch = 0
    quant_epoch = 5
    bp_iter = 200


num_weight_centers_encoder = 16
num_weight_centers_decoder = 8
encoder_sparsity = 0.4
decoder_sparsity = 0.8


"""
Load & prepare a dataset and a model
"""
opt = TestOptions().parse(save=False)
opt.nThreads = 2  # test code only supports nThreads = 1
opt.batchSize = 128  # test code only supports batchSize = 1
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
device = torch.device("cuda") if len(opt.gpu_ids)>0 else torch.device("cpu")

criterion = torch.nn.MSELoss().to(device)

model = model.to(device)

"""
Fuse Conv + BatchNorm Layers
"""
if not opt.fuse_layers: # opt.fuse_layers=False means that the layers are not fused yet. So we will fuse Conv & BN layers.
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


"""
Prepare & set the Activation quantization
This will increase the performance of the model after converting it into the 8-bit representation.
""" 

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

# Observation is required to use 'quant' mode in the activation modules
for m in model.modules():
    m.mode = 'observe'
    m.quantize_gradient=False
evaluate()


# Now set the activations to the 'quant' mode. The activations is now converted to the fake 8-bit representation.
for m in model.modules():
    if isinstance(m, FixedReLU) or isinstance(m, FixedInputQuantizer) or isinstance(m, FixedSigmoid):
        m.mode = 'quant'

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


"""
Clamp parameters
Some weights have extremely large values because of Conv+BN fusion, so they need to be clipped. 
"""
for p in total_params:
    with torch.no_grad():
        p.data = p.clamp_(-1.0, 1.0)

"""
Train weights to reduce the impact of the Conv+BN fusion and the weight clipping.
"""
optmize_Com = torch.optim.Adam(total_params, lr=lr, weight_decay=0e-3)
for e in range(first_pretrain_epoch):
    for k, data in enumerate(dataset):
        input_label, image = model.encode_input(Variable(data['label']), Variable(data['image']), infer=True)

        msssim_loss = 1
        optmize_Com.zero_grad()
        generated_img, _ = model.inference(input_label, "None")
        mse_loss = criterion(generated_img, input_label)
        if k % 200 == 0:#j % 50 == 49 or j==0:
            print('{:4d} MSE: {:0.4f} '.format(k+1, mse_loss.item()))
        mse_loss.backward()
        optmize_Com.step()




"""
Set sparsity of each layer
"""

modules = []
conv_modules = []
module_idx = 0
for name, module in model.named_modules():
    if isinstance(module, FixedConvTranspose2d) or isinstance(module, FixedConv2d):
        conv_modules.append(module)
        if module_idx >= 7 : # 0~5: Encoder, 6~: Decoder
            module.num_weight_centers = num_weight_centers_decoder
            module.sparsity = decoder_sparsity
        else:
            module.num_weight_centers = num_weight_centers_encoder
            module.sparsity = encoder_sparsity
        module_idx += 1

"""
Prune weights
"""

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




for module in conv_modules:
    with torch.no_grad():
        module.weight_mask_ = get_mask(module.weight, module.sparsity)
        prune.custom_from_mask(module, 'weight', module.weight_mask_)


"""
Retraining Stage
"""
for e in range(second_pretrain_epoch):
    for k, data in enumerate(dataset):
        input_label, image = model.encode_input(Variable(data['label']), Variable(data['image']), infer=True)

        msssim_loss = 1
        optmize_Com.zero_grad()
        generated_img, _ = model.inference(input_label, "None")
        mse_loss = criterion(generated_img, input_label)
        if k % 200 == 0:#j % 50 == 49 or j==0:
            print('{:4d} MSE: {:0.4f} '.format(k+1, mse_loss.item()))
        mse_loss.backward()
        optmize_Com.step()



for module in conv_modules:
    prune.remove(module, 'weight')
    module.weight_mask = module.weight_mask_




for name, module in model.named_modules():
    if isinstance(module, FixedConvTranspose2d) or isinstance(module, FixedConv2d):
        print(name, torch.sum(module.weight == 0).item()/float(module.weight.nelement()))



save_dir = os.path.join(opt.checkpoints_dir, opt.name)
if not first_pretrain_epoch and not second_pretrain_epoch and os.path.exists(os.path.join(save_dir, 'deepcomp_pruned_{:.2f}_{:.2f}_net_E.pth'.format(encoder_sparsity, decoder_sparsity))):
    model.load_network(model.netE, 'E', 'deepcomp_pruned_{:.2f}_{:.2f}'.format(encoder_sparsity, decoder_sparsity), './checkpoints/timit_small_relu4_final/')
    model.load_network(model.netDecoder, 'Decoder', 'deepcomp_pruned_{:.2f}_{:.2f}'.format(encoder_sparsity, decoder_sparsity), './checkpoints/timit_small_relu4_final/')
    print("Pruned net loaded")

elif not test_mode and not first_pretrain_epoch and not second_pretrain_epoch:
    model.save('deepcomp_pruned_{:.2f}_{:.2f}'.format(encoder_sparsity, decoder_sparsity), save_netD=False)



model.to(device)


"""
Quantization Stage
"""

def get_cluster_index(weight, centers):
    assert len(weight.size()) == 4 or len(weight.size()) == 2
    with torch.no_grad():
        w = weight.view(weight.size()+(1,))
        centers = centers.view((1,1,1,1,-1)) if len(weight.size()) == 4 else centers.view((1,1,-1))
        idx = torch.argmin(torch.abs(w - centers), dim=-1)
    return idx
    

for i, module in reversed(list(enumerate(conv_modules))):
    print("{:2d}/{:2d}".format(len(conv_modules) - i , len(conv_modules)))
    num_weight_centers = module.num_weight_centers
    weight = module.weight
    with torch.no_grad():
        """
        Find centroids
        """
        space = torch.linspace(weight.min(), weight.max(), steps=num_weight_centers)

        kmeans = KMeans(n_clusters = len(space), init=space.reshape(-1, 1), n_init=1)
        kmeans.fit(weight.clone().detach().cpu().numpy().reshape(-1, 1))
        centers = kmeans.cluster_centers_.reshape(-1)

        centers = np.unique(centers)
        module.num_weight_centers = len(centers)
        centers = torch.tensor(centers)

        if weight.is_cuda:
            centers = centers.cuda()
        module.centers = centers
        module.cluster_idx = get_cluster_index(module.weight, module.centers)
        module.weight.data = module.centers[module.cluster_idx]

    params_list = []
    for j in range(i):
        params_list += list(conv_modules[j].parameters())
    params_list += [conv_modules[i].bias]
    optimizer = torch.optim.Adam(params_list, lr=lr)
    for e in range(quant_epoch):
        for k, data in enumerate(dataset):
            input_label, image = model.encode_input(Variable(data['label']), Variable(data['image']), infer=True)
            generated_img, _ = model.inference(input_label, "None")
            mse_loss = criterion(generated_img, input_label)
            if k % 200 == 0:#j % 50 == 49 or j==0:
                print('{:4d} MSE: {:0.4f} '.format(k+1, mse_loss.item()))
            mse_loss.backward()
            with torch.no_grad():
                for idx in range(module.num_weight_centers):
                    module.centers[idx] -= lr * torch.mean(module.weight.grad[module.cluster_idx == idx])
                module.weight.data = module.weight_mask * module.centers[module.cluster_idx]

            
            optimizer.step()
            optimizer.zero_grad()
            module.weight.grad = None
                

    with torch.no_grad():
        n_fractional = 7
        weight = (module.weight * (2**n_fractional)).type(torch.long).type(torch.float32)
        weight = torch.clamp_(weight, -2**n_fractional+1, 2**n_fractional-1)
        module.weight.data = weight / (2**n_fractional)

        bias = (module.bias * (2**n_fractional)).type(torch.long).type(torch.float32)
        bias = torch.clamp_(bias, -2**n_fractional+1, 2**n_fractional-1)
        module.bias.data = bias / (2**n_fractional)

    for k, data in enumerate(dataset):
        if k >= bp_iter:
            break
        input_label, image = model.encode_input(Variable(data['label']), Variable(data['image']), infer=True)
        generated_img, _ = model.inference(input_label, "None")
        mse_loss = criterion(generated_img, input_label)
        mse_loss.backward()
                    
        optimizer.step()
        optimizer.zero_grad()
         


    with torch.no_grad():
        generated_img, _ = model.inference(test_image, "Hard")
        gen_img = util.tensor2im(generated_img[0])
        mse_loss = criterion(generated_img[:,:,:test_image.size(2),:test_image.size(3)], test_image)
        print("Test MSE:", mse_loss.item())



model.save('deepcomp_Q_pruned', save_netD=False)