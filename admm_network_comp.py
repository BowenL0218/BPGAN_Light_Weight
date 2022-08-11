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
num_bias_centers = 128
lr = 0.0002
alpha = 10
mu = 0.01
if test_mode:
    first_pretrain_epoch = 0
    second_pretrain_epoch = 0
    how_many_infer = 2
    ADMM_iter_Q = 2
    ADMM_iter_C = 2
    bp_iter = 10
else:
    first_pretrain_epoch = 1
    second_pretrain_epoch = 1
    how_many_infer = 100
    ADMM_iter_Q = 20
    ADMM_iter_C = 20
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
if not opt.fused_layers: # opt.fused_layers=False means that the layers are not fused yet. So it will fuse Conv & BN layers.
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
full_modules = []
module_idx = 0
for name, module in model.named_modules():
    if isinstance(module, FixedConvTranspose2d) or isinstance(module, FixedConv2d):
        full_modules.append(module)
        if module_idx >= 7 : # 0~5: Encoder, 6~: Decoder
            module.num_weight_centers = num_weight_centers_decoder
            module.sparsity = decoder_sparsity
        else:
            module.num_weight_centers = num_weight_centers_encoder
            module.sparsity = encoder_sparsity
        module_idx += 1

model.train()


def project_sparse_weight(weight, sparsity):
    """
    Prune weights
    """
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



"""
ADMM Pruning Stage
"""
optmize_Com = torch.optim.Adam(total_params, lr=lr, weight_decay=0e-2)
for e in range(second_pretrain_epoch):
    for k, data in enumerate(dataset):
        input_label, image = model.encode_input(Variable(data['label']), Variable(data['image']), infer=True)

        for module in full_modules:
            with torch.no_grad():
                # Initialize Z & eta
                module.Z = project_sparse_weight(module.weight, module.sparsity)
                module.eta = torch.zeros_like(module.weight).cuda()

        for j in range(ADMM_iter_C):
            optmize_Com.zero_grad()
            generated_img, _ = model.inference(input_label, "None")
            mse_loss = criterion(generated_img, input_label)
            Com_loss =  alpha * mse_loss +  mu / 2.0 * \
                            torch.mean(torch.stack([torch.norm(module.weight - module.Z + module.eta, 2) ** 2 \
                            for module in full_modules]))
            if k % 10 == 0 and j % 10 == 0:
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
if not first_pretrain_epoch and not second_pretrain_epoch and os.path.exists(os.path.join(save_dir, 'pruned_{:.2f}_{:.2f}_net_E.pth'.format(encoder_sparsity, decoder_sparsity))):
    model.load_network(model.netE, 'E', 'pruned_{:.2f}_{:.2f}'.format(encoder_sparsity, decoder_sparsity), './checkpoints/timit_small_relu4_final_2/')
    model.load_network(model.netDecoder, 'Decoder', 'pruned_{:.2f}_{:.2f}'.format(encoder_sparsity, decoder_sparsity), './checkpoints/timit_small_relu4_final_2/')
    print("Pruned net loaded")

elif not test_mode:
    model.save('pruned_{:.2f}_{:.2f}'.format(encoder_sparsity, decoder_sparsity), save_netD=False)



model.to(device)


"""
ADMM Quantization Stage
"""

for e in range(1):
    for i, weight in reversed(list(enumerate(total_params))):
        print(len(total_params)-i,"/",len(total_params))

        num_weight_centers = full_modules[i//2].num_weight_centers if len(weight.size())!=1 else num_bias_centers

        is_bias = len(weight.size()) == 1

        if len(weight.view(-1)) > num_weight_centers:
            with torch.no_grad():
                """
                Find centroids
                """
                w = weight.data.detach().clone().cpu().numpy().reshape(-1, 1)
                kmeans = KMeans(n_clusters=min(num_weight_centers, len(np.unique(w))))
                centers = kmeans.fit(w).cluster_centers_.reshape(-1)
                        
                n_fractional = 7#get_fractional(centers) 
                centers = (centers * (2**n_fractional)).astype("int32").astype("float32") #/ (2**n_fractional)
                centers = np.clip(centers, -2**n_fractional+1, 2**n_fractional-1)
                centers = centers / (2**n_fractional)
                centers = np.unique(centers)
                centers = centers.reshape(1, -1)
                print('Num Centers:' ,len(centers.reshape(-1)), centers.max(), centers.min())
                centers = torch.Tensor(centers).cuda()
                    
            optmize_Com = torch.optim.Adam(total_params[:i+1], lr=lr, weight_decay=0e-2)

            for k, data in enumerate(dataset):
                if k >= how_many_infer:
                    break
                input_label, image = model.encode_input(Variable(data['label']), Variable(data['image']), infer=True)
                weight_clone = weight.clone().detach().requires_grad_()

                if len(opt.gpu_ids) > 0:
                    weight = weight.cuda()
                with torch.no_grad():
                    Z = nonlinear_quantize(weight_clone.data, centers)
                    eta = torch.zeros(weight_clone.shape)
                if len(opt.gpu_ids)>0:
                    eta = eta.cuda()
                msssim_loss = 1
                for j in range(ADMM_iter_Q):
                    optmize_Com.zero_grad()
                    generated_img, _ = model.inference(input_label, "None")
                    mse_loss = criterion(generated_img, input_label)
                    feat_loss = torch.tensor(0.).to(device)

                    ssim_loss = 1 - ssim((generated_img+1)/2, (input_label+1)/2, data_range=1)
                    Com_loss =  alpha * (mse_loss + 0.1*feat_loss) +  mu / 2.0 * torch.norm(weight - Z + eta, 2) ** 2 
                    if k%50==0 and j % 10 == 0:
                        print('{:3d} {:3d} MSE: {:0.4f} SSIM Loss: {:0.4f} Feature Loss: {:0.4f} ADMM Loss: {:0.4f}'.format(k+1, j+1, mse_loss.item(), ssim_loss.item(), feat_loss.item(), Com_loss.item())) 
                            
                    Com_loss.backward()
                    optmize_Com.step()
                    with torch.no_grad():
                        weight_clone.data = weight.data
                        Z = nonlinear_quantize(weight_clone.data + eta, centers)
                        eta = eta + weight.data - Z
                with torch.no_grad():
                    if is_bias:
                        weight.data =  nonlinear_quantize(weight_clone.data, centers)
                    else:
                        weight.data = full_modules[i//2].weight_mask * nonlinear_quantize(weight_clone.data, centers)
        else:
            w = weight.data.detach().clone().cpu().numpy()
            n_fractional = get_fractional(w)
            with torch.no_grad():
                if is_bias:
                    weight.data = torch.Tensor((w * (2**n_fractional)).astype("int32").astype("float32") / (2**n_fractional)).cuda()
                else:
                    weight.data = full_modules[i//2].weight_mask * torch.Tensor((w * (2**n_fractional)).astype("int32").astype("float32") / (2**n_fractional)).cuda()

        weight.cuda()


        print('Sparsity', torch.sum(weight == 0).item()/float(weight.nelement()))

        model.save('ADMM_Q_{:02d}'.format(len(total_params)-i), save_netD=False)



        """
        Train the rest of layers once again.
        """
        if i > 0:
            optmize_Com = torch.optim.Adam(total_params[:i], lr=lr, weight_decay=0e-2)
            for k, data in enumerate(dataset):
                if k >= bp_iter:
                    break
                input_label, image = model.encode_input(Variable(data['label']), Variable(data['image']), infer=True)

                msssim_loss = 1
                optmize_Com.zero_grad()
                generated_img, _ = model.inference(input_label, "None")
                mse_loss = criterion(generated_img, input_label)

                l1_loss = 0e-2 *  torch.mean(torch.cat([torch.abs(p).view(-1) for p in total_params[:i]], dim=0))
                Com_loss =  alpha * mse_loss + l1_loss


                Com_loss.backward()
                optmize_Com.step()

        with torch.no_grad():
            generated_img, _ = model.inference(test_image, "Hard")
            gen_img = util.tensor2im(generated_img[0])
            mse_loss = criterion(generated_img[:,:,:test_image.size(2),:test_image.size(3)], test_image)
            print("Test MSE:", mse_loss.item())
        imageio.imwrite(os.path.join(output_path, '{:02d}.png'.format(len(total_params)-i)), gen_img)



for i, weight in reversed(list(enumerate(total_params))):
    if len(weight.size()) != 1:
        prune.remove(full_modules[i//2], 'weight')


model.save('ADMM_Q_pruned', save_netD=False)
