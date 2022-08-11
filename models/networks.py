import torch
import torchvision
import torch.nn as nn
import functools
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from .custom_layer import custom_BatchNorm2d, custom_BN
from .Bpgan_VGG_Extractor import Bpgan_VGGLoss
#from .quantization_modules import FixedConvTranspose,FixedLinear,FixedConv2d
from .fixed_point_modules import FixedConv2d, FixedConvTranspose2d, FixedInputQuantizer, FixedAdd, FixedBatchNorm2d, FixedReLU, FixedSigmoid, FixedTanh

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_uniform_(m.weight)
    elif classname.find('ConvTranspose2d') != -1:
        init.kaiming_uniform_(m.weight)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'cus':
        norm_layer = functools.partial(custom_BatchNorm2d,affine=True,track_running_stats=True)
    elif norm_type == 'cus_2':
        norm_layer = functools.partial(custom_BN)
    elif norm_type in ['none', 'identity']:
        norm_layer = nn.Identity
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_E(input_nc,ngf,n_downsample=3,C_channel=8,norm='instance', gpu_ids=[],one_D_conv=False, one_D_conv_size=63, max_ngf=512,Conv_type = "C", pool_type='none', noisy_activation=False, fuse_layers=False):
    norm_layer = get_norm_layer(norm_type=norm)
    if input_nc == 1:
        netE = Encoder(input_nc=input_nc,ngf=ngf,C_channel=C_channel,n_downsampling=n_downsample,norm_layer=norm_layer,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size, max_ngf=max_ngf, Conv_type= Conv_type, pool_type=pool_type, noisy_activation=noisy_activation)
    else:
        netE = EncoderV2(input_nc=input_nc,ngf=ngf,C_channel=C_channel,n_downsampling=n_downsample,norm_layer=norm_layer,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size, max_ngf=max_ngf, Conv_type= Conv_type, pool_type=pool_type, noisy_activation=noisy_activation)
    if len(gpu_ids) >0:
        assert (torch.cuda.is_available())
        netE.cuda(gpu_ids[0])
    if fuse_layers:
        for m in netE.modules():
            if isinstance(m, ConvBNReLU) or isinstance(m, ConvTransposeBNReLU):
                m.fuse_layers()
    netE.apply(weights_init)
    return netE

def define_Decoder(output_nc,ngf,n_downsample=3,C_channel=8,n_blocks_global=9,norm="instance",gpu_ids=[],one_D_conv=False, one_D_conv_size=63, max_ngf = 512, Conv_type="C",Dw_Index=None, noisy_activation=False, fuse_layers=False):
    if output_nc == 3:
        Conv_type="C"
    norm_layer = get_norm_layer(norm_type=norm)
    if output_nc == 1:
        netDecoder = Decoder(ngf=ngf,C_channel=C_channel,n_downsampling=n_downsample,output_nc=output_nc,n_blocks=n_blocks_global,norm_layer=norm_layer,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size, max_ngf=max_ngf, Conv_type = Conv_type, Dw_Index=Dw_Index, noisy_activation=noisy_activation)
    else:
        netDecoder = DecoderV2(ngf=ngf,C_channel=C_channel,n_downsampling=n_downsample,output_nc=output_nc,n_blocks=n_blocks_global,norm_layer=norm_layer,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size, max_ngf=max_ngf, Conv_type = Conv_type, Dw_Index=Dw_Index, noisy_activation=noisy_activation)

    if len(gpu_ids) >0:
        assert (torch.cuda.is_available())
        netDecoder.cuda(gpu_ids[0])

    if fuse_layers:
        for m in netDecoder.modules():
            if isinstance(m, ConvBNReLU) or isinstance(m, ConvTransposeBNReLU):
                m.fuse_layers()
    netDecoder.apply(weights_init)
    return netDecoder


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[],one_D_conv=False, one_D_conv_size=63):
    norm_layer = get_norm_layer(norm_type=norm)
    if input_nc == 1:
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size)
    else:
        netD = ImageDiscriminator()#input_nc, ndf, n_layers_D, norm_layer)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)



##############################################################################
# Generator
##############################################################################


def fuse_conv_bn(conv, bn):
    fused_conv = copy.deepcopy(conv)
    fused_conv.weight, fused_conv.bias = \
            nn.utils.fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                    bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return fused_conv


def fuse_convTranspose_bn(conv, bn):
    fused_conv = copy.deepcopy(conv)
    print(fused_conv.weight.size())
    weight, fused_conv.bias = \
            nn.utils.fuse_conv_bn_weights(fused_conv.weight.permute(1,0,2,3), fused_conv.bias,
                    bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    fused_conv.weight.data = weight.permute(1,0,2,3)
    print(fused_conv.weight.size())
    return fused_conv


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, 
                       out_channels,
                       kernel_size, 
                       stride=1, 
                       groups=1, 
                       padding_mode='zeros', 
                       no_activation=False, 
                       no_bn=False,
                       noisy_activation=False):
        padding = (kernel_size-1)//2
        self.no_activation = no_activation
        layers = [FixedConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=padding_mode)]
        if not no_bn:
            layers += [FixedBatchNorm2d(out_channels)]

        if not self.no_activation:
            layers += [FixedReLU(noisy_activation=noisy_activation)]
        super(ConvBNReLU, self).__init__(*layers)

    def fuse_layers(self):
        fused_layer = fuse_conv_bn(self[0], self[1])
        self[0] = fused_layer
        self[1] = nn.Identity()


class ConvTransposeBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, padding_mode='zeros', no_activation=False, noisy_activation=False):
        padding = (kernel_size-1)//2
        self.no_activation = no_activation
        layers = [FixedConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=padding, groups=groups, padding_mode=padding_mode),
            FixedBatchNorm2d(out_channels)]
        if not self.no_activation:
            layers += [FixedReLU(noisy_activation=noisy_activation)]
        super(ConvTransposeBNReLU, self).__init__(*layers)

    def fuse_layers(self):
        fused_layer = fuse_convTranspose_bn(self[0], self[1])
        self[0] = fused_layer
        self[1] = nn.Identity()



class Encoder(nn.Module):
    def __init__(self,input_nc,ngf=64,C_channel=8, n_downsampling=3,norm_layer=nn.BatchNorm2d,padd_type="reflect",one_D_conv=False, one_D_conv_size=63,max_ngf=512, Conv_type="C", pool_type='none', n_blocks=5, noisy_activation=False):
        assert(n_downsampling>=0)
        super(Encoder, self).__init__()
        activation = FixedReLU(noisy_activation=noisy_activation)
        model = []

        if one_D_conv:
            model += [nn.ReflectionPad2d((0,0,(one_D_conv_size-1)//2,(one_D_conv_size-1)//2)),my_Conv(input_nc,int(ngf/2),kernel_size=(one_D_conv_size,1),type=Conv_type, norm_layer=norm_layer, activation=activation),norm_layer(ngf),activation,
                    nn.ReflectionPad2d(3), my_Conv(int(ngf/2),ngf, kernel_size=7, padding=0,type=Conv_type, norm_layer=norm_layer, activation=activation), norm_layer(ngf),activation ]
        else:
            model += [ConvBNReLU(input_nc, ngf, kernel_size=7, padding_mode='zeros', stride=1 if Conv_type=="C" else 2, noisy_activation=noisy_activation)]
        if Conv_type == "E":
            n_downsampling -= 1
        ##downsample
        for i in range(n_downsampling):
            mult = 2**i
            if "C" in Conv_type:
                model += [ConvBNReLU(min(ngf * mult, max_ngf), min(ngf*mult*2, max_ngf), kernel_size=3, stride=2, noisy_activation=noisy_activation)]
            else: # DW+1x1
                model += [ConvBNReLU(min(ngf * mult, max_ngf), min(ngf*mult, max_ngf), kernel_size=3, stride=2, groups=min(ngf * mult, max_ngf), noisy_activation=noisy_activation)]
                model += [ConvBNReLU(min(ngf * mult, max_ngf), min(ngf*mult*2, max_ngf), kernel_size=1, stride=1, groups=1, noisy_activation=noisy_activation)]

        self.model = nn.Sequential(*model)
        self.projection = nn.Sequential(ConvBNReLU(min(ngf * (2 ** n_downsampling), max_ngf), C_channel, kernel_size=3, no_activation=True), FixedSigmoid())
        self.quant = FixedInputQuantizer()

    def forward(self, x):
        z = self.quant(x)
        z =  self.model(z)
        z = self.projection(z)
        return z

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                #torch.quantization.fuse_modules(m, ['0', '1'] if m.no_activation else ['0', '1', '2'], inplace=True)
                m.fuse_layers()


class EncoderV2(nn.Module):
    def __init__(self,input_nc,ngf=64,C_channel=8, n_downsampling=3,norm_layer=nn.BatchNorm2d,padd_type="reflect",one_D_conv=False, one_D_conv_size=63,max_ngf=512, Conv_type="C", pool_type='none', n_blocks=5, noisy_activation=False):
        super(EncoderV2, self).__init__()
        assert(n_downsampling>=0)
        model = []

        model += [ConvBNReLU(input_nc, ngf, kernel_size=7, padding_mode='zeros', stride=1 if Conv_type=="C" else 2, noisy_activation=noisy_activation, no_bn=True)]
        ##downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [ConvBNReLU(min(ngf * mult, max_ngf), min(ngf*mult*2, max_ngf), kernel_size=3, stride=2, noisy_activation=noisy_activation, no_bn=True)]
            #model += [ConvBNReLU(min(ngf*mult*2, max_ngf), min(ngf*mult*2, max_ngf), kernel_size=3, stride=1, noisy_activation=noisy_activation, no_bn=True)]

        self.model = nn.Sequential(*model)
        self.projection = nn.Sequential(ConvBNReLU(min(ngf * (2 ** n_downsampling), max_ngf), C_channel, kernel_size=3, no_activation=True, no_bn=True), FixedSigmoid())
        self.quant = FixedInputQuantizer()
        self.quant.mode = 'quant' 
        #self.maxpool = nn.MaxPool2d(2,2)
        self.n_downsampling = n_downsampling

    def forward(self, x):
        x = self.quant(x)
        #x_pool = self.maxpool(x)
        #if self.n_downsampling == 2:
        #    x_pool = self.maxpool(x_pool)
        x =  self.model(x)
        x = self.projection(x)

        return x



class Decoder(nn.Module):
    def __init__(self,ngf=64,C_channel=8, n_downsampling=3,output_nc=1,n_blocks=9, norm_layer=nn.BatchNorm2d,padding_type="reflect",one_D_conv=False, one_D_conv_size=63,max_ngf=512, Conv_type="C", Dw_Index=None, noisy_activation=False):
        assert (n_blocks>=0)
        super(Decoder, self).__init__()
        activation = FixedReLU() if output_nc == 1 else nn.LeakyReLU()
        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        res_model = [ConvBNReLU(C_channel, ngf_dim, kernel_size=3, noisy_activation=noisy_activation)]
        for i in range(n_blocks):
            res_model += [ResnetBlock(ngf_dim, Conv_type=Conv_type, noisy_activation=noisy_activation)]

        upsample_layers = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i) 
            if Conv_type == "C":# or i == n_downsampling-1:
                upsample_layers += [ConvTransposeBNReLU(min(ngf * mult, max_ngf), min(ngf * mult // 2, max_ngf), kernel_size=3, stride=2, noisy_activation=noisy_activation)]
            elif Conv_type == "E":
                upsample_layers += [ConvBNReLU(min(ngf*mult, max_ngf), min(ngf*mult//4, max_ngf), kernel_size=1, noisy_activation=noisy_activation),
                                    ConvTransposeBNReLU(min(ngf*mult//4, max_ngf), min(ngf*mult//4, max_ngf), kernel_size=3, stride=2, noisy_activation=noisy_activation),
                                    ConvBNReLU( min(ngf*mult//4, max_ngf), min(ngf*mult//2, max_ngf),kernel_size=1, noisy_activation=noisy_activation)]
                                    
        if one_D_conv:
            final_layer = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, int(ngf/2), kernel_size=7, padding=0),activation,norm_layer(int(ngf/2)),
                      nn.ReflectionPad2d((0,0,(one_D_conv_size-1)//2,(one_D_conv_size-1)//2)),nn.Conv2d(int(ngf/2),output_nc,kernel_size=(one_D_conv_size,1),padding=0),nn.Tanh()]
        else:
            final_layer = [ConvBNReLU(ngf, output_nc, kernel_size=7, no_activation=True), FixedTanh()]

        self.res_model = nn.Sequential(*res_model)
        self.upsample_layers = nn.Sequential(*upsample_layers)
        self.final_layer = nn.Sequential(*final_layer)
        self.quant = FixedInputQuantizer()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.res_model(x)
        x = self.upsample_layers(x)
        x = self.final_layer(x)
        return x

    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                #torch.quantization.fuse_modules(m, ['0', '1'] if m.no_activation else ['0', '1', '2'], inplace=True)
                m.fuse_layers()


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.qconv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.kconv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.vconv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        b, c, h, w = x.size()
        
        query = self.qconv(x)
        key = self.kconv(x)
        value = self.vconv(x)

        query = torch.flatten(query, start_dim=2)
        key = torch.flatten(key, start_dim=2)
        value = torch.flatten(value, start_dim=2)

        energy = torch.bmm(query.permute(0,2,1), key)
        attention_map = torch.softmax(energy, dim=-1)
        
        out = torch.bmm(value, attention_map.permute(0,2,1)).view(b,c,h,w)
        out = self.gamma * out + x
        return out


class UpsampleBlock(nn.Module):
    """Some Information about UpsampleBlock"""
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        #self.conv = nn.Conv2d(in_channels, out_channels*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.upsample = nn.PixelShuffle(2)
        #self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        #x = self.upsample(x)
        x = self.bn(x)
        x = nn.LeakyReLU(0.2)(x)
        #x = self.prelu(x)

        return x

class DecoderV2(nn.Module):
    def __init__(self,ngf=64,C_channel=8, n_downsampling=3,output_nc=1,n_blocks=9, norm_layer=nn.BatchNorm2d,padding_type="reflect",one_D_conv=False, one_D_conv_size=63,max_ngf=512, Conv_type="C", Dw_Index=None, noisy_activation=False):
        super(DecoderV2, self).__init__()
        assert (n_blocks>=0)
        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        self.first_conv = nn.Sequential(ConvBNReLU(C_channel, ngf_dim, kernel_size=3, noisy_activation=noisy_activation, no_activation=True, no_bn=True), nn.ReLU())#, SEBlock(ngf_dim, 2))
        res_model = []
        for i in range(n_blocks):
            res_model += [ResnetBlock(ngf_dim, Conv_type=Conv_type, noisy_activation=noisy_activation)]

        upsample_layers = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i) 
            upsample_layers += [UpsampleBlock(min(ngf * mult, max_ngf), min(ngf * mult // 2, max_ngf))]
                                #SEBlock(min(ngf * mult // 2, max_ngf), 2)]
            #upsample_layers += [ResnetBlock(min(ngf * mult // 2, max_ngf), Conv_type="C")]

                                    
        final_layer = [ConvBNReLU(ngf, output_nc, kernel_size=9, no_activation=True), nn.Tanh()]

        self.res_model = nn.Sequential(*res_model)
        self.upsample_layers = nn.Sequential(*upsample_layers)
        self.final_layer = nn.Sequential(*final_layer)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.res_model(x) + x
        x = self.upsample_layers(x)
        x = self.final_layer(x)
        return x



class quantizer(nn.Module):
    def __init__(self,center,Temp):
        super(quantizer, self).__init__()
        self.center = nn.Parameter(center)
        self.register_parameter('center',self.center)
        self.Temp = Temp
        self.code_indices = []
    def forward(self, x, Q_type="None", store_code_idx=False):
        if Q_type=="Soft":
            W_stack = torch.stack([x for _ in range(len(self.center))],dim=-1)
            W_index = torch.argmin(torch.abs(W_stack-self.center),dim=-1)
            W_hard = self.center[W_index]
            smx = torch.softmax(-1.0*self.Temp*(W_stack-self.center)**2,dim=-1)
            W_soft = torch.einsum('ijklm,m->ijkl',[smx,self.center])
            with torch.no_grad():
                w_bias = (W_hard - W_soft)
            return w_bias + W_soft
        elif Q_type=='None':
            return x
        elif Q_type == 'Hard':
            W_stack = torch.stack([x for _ in range(len(self.center))], dim=-1)
            #W_index = torch.argmin(torch.abs(W_stack - self.center), dim=-1)
            W_index = torch.argmin(torch.abs(W_stack - self.center), dim=-1)
            W_hard = self.center[W_index]
            if store_code_idx:
                self.code_indices.append(W_index.detach().cpu().numpy().reshape(-1))
            return W_hard

    def get_indices(self, x):
        W_stack = torch.stack([x for _ in range(len(self.center))], dim=-1)
        #W_index = torch.argmin(torch.abs(W_stack - self.center), dim=-1)
        W_index = torch.argmin(torch.abs(W_stack - self.center), dim=-1)
        return W_index
    def update_Temp(self,new_temp):
        self.Temp = new_temp
    def update_center(self,new_center):
        self.center = nn.Parameter(new_center)



class vector_quantizer(nn.Module):
    def __init__(self,center,Temp):
        super(vector_quantizer, self).__init__()
        self.center = nn.Parameter(center)
        self.register_parameter('center',self.center)
        self.Temp = Temp
        self.code_indices = []
    def forward(self, x,Q_type='None', store_code_idx=False):
        x_ = x.view(x.shape[0],-1,4)
        if Q_type=="Soft":
            sim = -torch.sum((x_.view(x.shape[0], -1, 1, 4) - self.center)**2, dim=-1)
            W_index = torch.argmax(sim,dim=-1)
            W_hard = self.center[W_index]
            smx = torch.softmax(self.Temp*sim, dim=-1)
            output = torch.matmul(smx, self.center)
        elif Q_type=='None':
            output = x_
        elif Q_type == 'Hard':
            sim = -torch.sum((x_.view(x.shape[0], -1, 1, 4) - self.center)**2, dim=-1)
            W_index = torch.argmax(sim,dim=-1)
            W_hard = self.center[W_index]
            output =  W_hard
            if store_code_idx:
                self.code_indices.append(W_index.detach().cpu().numpy().reshape(-1))
        return output.view(x.shape)
    def update_Temp(self,new_temp):
        self.Temp = new_temp
    def update_center(self,new_center):
        self.center = nn.Parameter(new_center)

    def get_indices(self, x):
        x_ = x.view(x.shape[0],-1,4)
        W_stack = torch.stack([x_ for _ in range(len(self.center))], dim=-1)
        E = torch.norm(W_stack - self.center.transpose(1, 0), 2, dim=-2)
        W_index = torch.argmin(E, dim=-1)
        return W_index

class ResnetBlock(nn.Module):
    def __init__(self, dim, activation=nn.LeakyReLU(0.2), use_dropout=False, Conv_type="C", noisy_activation=False):
        super(ResnetBlock, self).__init__()
        self.activation = activation
        self.conv_block = self.build_conv_block(dim, Conv_type=Conv_type) 
        #self.skip_add = nn.quantized.FloatFunctional()
        self.skip_add = FixedAdd()
        self.noisy_activation=noisy_activation

    def build_conv_block(self, dim, Conv_type="C"):#, padding_type, norm_layer, activation, use_dropout, Conv_type="C"):
        conv_block = []
        p = 0
        if Conv_type == "C":
            conv_block += [ConvBNReLU(dim, dim, kernel_size=3, no_activation=True), 
                           nn.LeakyReLU(0.2),
                           ConvBNReLU(dim, dim, kernel_size=3, no_activation=True)]
                           #SEBlock(dim, 2)]
        else:
            conv_block += [ConvBNReLU(dim, dim//4, kernel_size=1, noisy_activation=self.noisy_activation)]
            conv_block += [ConvBNReLU(dim//4, dim//4, kernel_size=3, noisy_activation=self.noisy_activation)]
            conv_block += [ConvBNReLU(dim//4, dim, kernel_size=1, no_activation=True)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out 


class SEBlock(nn.Module):
    """Some Information about SEBlock"""
    def __init__(self, channels, r=2):
        super(SEBlock, self).__init__()
        self.senet = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Flatten(), 
                                     nn.Linear(channels, channels//r),
                                     nn.ReLU(), 
                                     nn.Linear(channels//r, channels),
                                     nn.Sigmoid())
                                     
    def forward(self, x):
        b, c, h, w = x.size()
        x_ = self.senet(x)
        x_ = x_.view(b, c, 1, 1)
        return x * x_


class ImageDiscriminator(nn.Module):
    """Some Information about ImageDiscriminator"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, 
                 **kwargs):
        super(ImageDiscriminator, self).__init__()

        layers = [nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2)]
        
        layers += [nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2)]
        for i in range(3):
            layers += [nn.Conv2d(64*(2**i), 128*(2**i), 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128*(2**i)), nn.LeakyReLU(0.2)]
            layers += [nn.Conv2d(128*(2**i), 128*(2**i), 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128*(2**i)), nn.LeakyReLU(0.2)]

        layers += [nn.AdaptiveAvgPool2d((6,6)), nn.Flatten(), nn.Linear(512*6*6, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, 1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return [[x]]


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False,one_D_conv=False, one_D_conv_size=63):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D-1):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)
        netD = NLayerDiscriminator(input_nc,ndf,n_layers,norm_layer,use_sigmoid,getIntermFeat,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size)
        if getIntermFeat:
            for j in range(n_layers+2):
                setattr(self, 'scale' + str(num_D-1) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
        else:
            setattr(self,'layer'+str(num_D-1),netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.one_D_conv = one_D_conv
    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input

        for i in range(num_D-1):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        if self.getIntermFeat:
            model = [getattr(self, 'scale' + str(num_D - 1) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
        else:
            model = getattr(self, 'layer' + str(num_D - 1))
        if self.one_D_conv:
            result.append(self.singleD_forward(model, input))
        else:
            result.append(self.singleD_forward(model,input_downsampled))
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False,one_D_conv=False, one_D_conv_size=63):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        if one_D_conv:
            sequence = [[nn.Conv2d(input_nc,int(ndf/2),kernel_size=(one_D_conv_size,1),padding=(0,(one_D_conv_size-1)//2)),nn.LeakyReLU(0.2,True),
                         nn.Conv2d(int(ndf/2),ndf,kernel_size=kw,stride=2,padding=padw),nn.LeakyReLU(0.2,True)]]
        else:
            sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

class Upsample2d(nn.Module):
    def __init__(self):
        super(Upsample2d, self).__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2.)


def my_Conv(nin, nout, kernel_size = 3, stride = 1, padding=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(), type="C", padding_mode='zeros'):
    if type == "C":
        return nn.Sequential(nn.Conv2d(nin,nout,kernel_size=kernel_size,stride=stride,padding=padding, padding_mode=padding_mode))
    elif type == "E" or "E_test":
        return nn.Sequential(
            nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, stride=stride, padding_mode=padding_mode),
            norm_layer(nin),
            activation,
            nn.Conv2d(nin,nout,kernel_size=1)
        )
    elif type == "FC":
        return nn.Sequential(FixedConv2d(nin,nout,kernel_size=kernel_size,stride=stride, padding=padding))
    elif type == "FE":
        return nn.Sequential(
            FixedConv2d(nin,nin,kernel_size=kernel_size,padding=padding,groups=nin,stride=stride),
            norm_layer(nin),
            activation,
            FixedConv2d(nin,nout,kernel_size=1)
        )
    else:
        raise NotImplementedError
def my_Deconv(nin, nout, norm_layer=nn.BatchNorm2d,activation=nn.ReLU(), type = "C"):
    if type == "C":
        return nn.Sequential(nn.ConvTranspose2d(nin, nout, kernel_size=3, stride=2, padding=1, output_padding=1))
    elif type == "E":
        return nn.Sequential(Upsample2d(), nn.Conv2d(nin//4, nout, kernel_size=3, padding=1))
        '''
        return nn.Sequential(
            upsample_pad(),
            nn.Conv2d(nin,nin,kernel_size=5,stride=1, padding=2,groups=nin),
            norm_layer(nin),
            activation,
            nn.Conv2d(nin, nout, kernel_size=1)
        )
        '''
    elif type == "E_test":
        return nn.Sequential(upsample_pad(),
                nn.Conv2d(nin, nout, kernel_size=3, stride=1, padding=1))
    elif type == "FC":
        return nn.Sequential(FixedConvTranspose(nin,out,kernel_size=3,stride=2,padding=1,output_padding=1))
    elif type == "FE":
        return nn.Sequential(
            upsample_pad(),
            FixedConv2d(nin, nin, kernel_size=3, stride=1, padding=1, groups=nin),
            norm_layer(nin),
            activation,
            FixedConv2d(nin, nout, kernel_size=1)
        )
    else:
        raise NotImplementedError

class upsample(nn.Module):
    def __init__(self):
        super(upsample, self).__init__()
        self.upsample =  lambda x: torch.nn.functional.interpolate(x, scale_factor=2)
    def forward(self, x):
        return self.upsample(x)
class upsample_pad(nn.Module):
    def __init__(self):
        super(upsample_pad, self).__init__()
    def forward(self,x):
        out = torch.zeros(x.shape[0],x.shape[1],2*x.shape[2],2*x.shape[3],device = x.device,dtype = x.dtype)
        out[:,:,0::2,:][:,:,:,0::2]=x
        return out


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer='2.2'):
        if layer != '2.2' and layer != '5.5' and layer != 'FinalMaxPool':
            raise ValueError('Invalid layer. Expected "2.2" or "5.5" or "FinalMaxPool", got: ', layer)
        super(VGGFeatureExtractor, self).__init__()
        VGG_temp = torchvision.models.vgg19(pretrained=True)
        # Add hook to the 2nd ReLU output of the 2nd block.
        if layer == '2.2':
            self.VGG = nn.Sequential(*list(VGG_temp.features.modules())[1:10])
        elif layer == '5.5': 
            self.VGG = nn.Sequential(*list(VGG_temp.features.modules())[1:33])
        elif layer == 'FinalMaxPool':
            self.VGG = nn.Sequential(*list(VGG_temp.features.modules())[1:])


    def forward(self, x):
        return self.VGG(x)


class Renormalize(nn.Module):
    def __init__(self, mean=torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1),
                       std=torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)):
        super(Renormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if x.is_cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        x = (x + 1) * 0.5
        return (x-self.mean)/self.std



class VGGFeatureLoss(nn.Module):
    def __init__(self, layer='5.5'):
        super(VGGFeatureLoss, self).__init__()
        #self.vggnet = VGGFeatureExtractor(layer)
        self.vggnet = TruncatedVGG19(2,2)
        self.vggnet.eval()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        #input = self.renormalize(input)
        input_feature = self.vggnet(input)
        target_feature = self.vggnet(target)#self.renormalize(target))
        return self.loss(input_feature, target_feature)



class TruncatedVGG19(nn.Module):
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network', as defined in the paper.
    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    """

    def __init__(self, i, j):
        """
        :param i: the index i in the definition above
        :param j: the index j in the definition above
        """
        super(TruncatedVGG19, self).__init__()

        # Load the pre-trained VGG19 available in torchvision
        vgg19 = torchvision.models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterate through the convolutional section ("features") of the VGG19
        for layer in vgg19.features.children():
            truncate_at += 1

            # Count the number of maxpool layers and the convolutional layers after each maxpool
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied
        assert maxpool_counter == i - 1 and conv_counter == j, "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (
            i, j)

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        Forward propagation
        :param input: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: the specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output
