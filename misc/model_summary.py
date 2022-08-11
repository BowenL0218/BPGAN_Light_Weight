import os
from collections import OrderedDict
from options.test_options import TestOptions
from options.train_Q_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TrainOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.model = 'Bpgan_GAN_Q'

#data_loader = CreateDataLoader(opt)
#dataset = data_loader.load_data()
model = create_model(opt)
model.eval()

#print(model.Q.center.size())
#z = torch.rand((1, 512, 40, 30))
#zQ = model.Q.forward(z, Q_type='Hard')
from torchsummary import summary
print("Encoder")
summary(model.netE, (3, 640, 480))
#summary(model.netE, (1, 64, 320))
print("Decoder")
summary(model.netDecoder, (opt.C_channel, 640//(2**opt.n_downsample_global), 480//(2**opt.n_downsample_global)))
#summary(model.netDecoder, (opt.C_channel, 64//(2**opt.n_downsample_global), 320//(2**opt.n_downsample_global)))
