import os
import torch
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from torch.quantization import QConfig, MinMaxObserver

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.model = 'Bpgan_GAN_Q'

eval_num=10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.to('cpu')
model.eval()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join('8bit', opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
def evaluate(name):
    for i, data in enumerate(dataset):
        if i >= eval_num:
            break
        if opt.model == 'Bpgan_GAN':
            generated, latent_vector = model.inference(data['label'])
        elif opt.model =='Bpgan_GAN_Q':
            generated, latent_vector = model.inference(data['label'],Q_type='Hard')
        elif opt.model == 'Bpgan_GAN_Q_Compressed':
            generated, latent_vector = model.inference(data['label'],Q_type='Hard')
        visuals = OrderedDict([(name, util.tensor2im(generated.data[0]))])
        img_path = data['path']
        #print('process image... %s' % img_path)
        #visualizer.save_images(webpage, visuals, img_path)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

#model.netE.fuse_model()
#model.netDecoder.fuse_model()

from torchsummary import summary
print("Encoder")
summary(model.netE, (1, 64, 480))
print("Decoder")
summary(model.netDecoder, (opt.C_channel, 64//(2**opt.n_downsample_global), 480//(2**opt.n_downsample_global)))
print("Before Compression")
print_size_of_model(model.netE)
print_size_of_model(model.netDecoder)

exit()
import math
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List, Tuple, Optional
class CustomMinMaxObserver(MinMaxObserver):
    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        if min_val.numel() == 0 or max_val.numel() == 0:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])

        if min_val.dim() == 0 or max_val.dim() == 0:
            assert min_val <= max_val, "min {} should be less than max {}".format(
                min_val, max_val
            )
        else:
            assert torch.sum(min_val <= max_val) == len(min_val), "min {} should be less than max {}".format(
                min_val, max_val
            )

        if self.dtype == torch.qint8:
            if self.reduce_range:
                qmin, qmax = -64, 63
            else:
                qmin, qmax = -128, 127
        else:
            if self.reduce_range:
                qmin, qmax = 0, 127
            else:
                qmin, qmax = 0, 255

        min_val = torch.min(min_val, torch.zeros_like(min_val))
        max_val = torch.max(max_val, torch.zeros_like(max_val))

        scale = torch.ones(min_val.size(), dtype=torch.float32)
        zero_point = torch.zeros(min_val.size(), dtype=torch.int64)
        device = 'cuda' if min_val.is_cuda else 'cpu'

        if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
            max_val = torch.max(-min_val, max_val)
            if max_val < 1.:
                scale = 1 / (float(qmax - qmin)/2)
            elif max_val >= 128:
                scale = 1.
            else:
                scale = max_val / (float(qmax - qmin) / 2)
                scale = torch.max(scale, torch.tensor(self.eps, device=device, dtype=scale.dtype))
            if self.dtype == torch.quint8:
                zero_point = zero_point.new_full(zero_point.size(), 128)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale = torch.max(scale, torch.tensor(self.eps, device=device, dtype=scale.dtype))
            zero_point = qmin - torch.round(min_val / scale)
            zero_point = torch.max(zero_point, torch.tensor(qmin, device=device, dtype=zero_point.dtype))
            zero_point = torch.min(zero_point, torch.tensor(qmax, device=device, dtype=zero_point.dtype))

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype)

        return scale, zero_point

qconfig = QConfig(activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric),
                  weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
model.netE.qconfig = qconfig
model.netDecoder.qconfig = qconfig
#model.Q.qconfig = qconfig
torch.quantization.prepare(model.netE, inplace=True)
torch.quantization.prepare(model.netDecoder, inplace=True)
#torch.quantization.prepare(model.Q, inplace=True)

evaluate('before')

#for m in model.netDecoder.modules():
#    print(m)
print(model.netDecoder)
exit()
netE_Q = torch.quantization.convert(model.netE, inplace=True)
netDecoder_Q = torch.quantization.convert(model.netDecoder, inplace=True)
#Q = torch.quantization.convert(model.Q, inplace=True)


model.netE = netE_Q
model.netDecoder = netDecoder_Q
#model.Q = Q

print("After Compression")
print_size_of_model(model.netE)
print_size_of_model(model.netDecoder)
evaluate('after')
#print(model.netE.model[1][0].weight())

print(model.netE)

print("saving model")
torch.jit.save(torch.jit.script(model.netE), "8bit/qmodel_E.pth")
torch.jit.save(torch.jit.script(model.netDecoder), "8bit/qmodel_D.pth")

netE = torch.jit.load("8bit/qmodel_E.pth")
netDecoder = torch.jit.load("8bit/qmodel_D.pth")
model.netE = netE
model.netDecoder = netDecoder

evaluate('after_sl')


    
