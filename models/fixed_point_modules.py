import torch
import torch.nn as nn
import torch.nn.functional as F

class QConfig:
    def __init__(self, act_type=None, weight_type=None):
        self.act_type = act_type
        self.weight_type = weight_type


class MinMaxObserver:
    def __init__(self, N=8):
        self.min_val = torch.Tensor([float("Inf")])
        self.max_val = torch.Tensor([-float("Inf")])
        self.scale = 1.0
        self.dtype = 'int8'
        self.N = N

    def observe(self, x, dtype, eps=1e-10):
        with torch.no_grad():
            self.dtype=dtype
            if not isinstance(x, torch.Tensor):
                raise TypeError("Invalid argumet type. Expected `torch.Tensor`, got: ", type(x))

            M, m = torch.max(x), torch.min(x)

            if x.is_cuda:
                self.min_val = self.min_val.cuda()
                self.max_val = self.max_val.cuda()

            if M > self.max_val:
                self.max_val = M

            if m < self.min_val:
                self.min_val = m

            self.scale = self.get_scale(dtype, eps)
            return self.scale

    def get_scale(self, dtype, eps=1e-10):
        with torch.no_grad():
            if dtype == 'uint8':
                max_val = torch.max(self.max_val, torch.zeros_like(self.max_val))
            elif dtype == 'int8':
                max_val = 2*torch.max(-self.min_val, self.max_val)
            
            scale = 2 ** (self.N - torch.ceil(torch.log2(max_val + eps)))
            return scale

        
        


def quantize(x, scale, dtype, N=8):
    with torch.no_grad():
        y = (x * scale).int().float()
        if dtype == 'uint8':
            y = torch.clamp(y, 0, 2**N-1)
            return y
        elif dtype == 'int8':
            y = torch.clamp(y, -2**(N-1), 2**(N-1)-1)
            return y
        else:
            raise NotImplementedError()

def dequantize(x, scale):
    with torch.no_grad():
        return x / scale

def convert_to_8bit(x, scale, dtype, N=8):
    return dequantize(quantize(x, scale, dtype, N), scale)


def add_gradient_observer_hook(observer):
    def hook(grad):
        observer.observe(grad, 'int8')
        return grad
    return hook


def add_gradient_quantizer_hook(observer):
    def hook(grad):
        grad.data = convert_to_8bit(grad, min(observer.scale*(2**3), 2**23), observer.dtype)
        return grad
    return hook


class FixedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        qconfig=None,
        mode='float',
        weight_observer=MinMaxObserver(),
        activation_observer=MinMaxObserver(),
        gradient_observer=MinMaxObserver()
    ): 

        super(FixedConv2d, self).__init__(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dilation,
                                        groups,
                                        bias,
                                        padding_mode)
        self.qconfig=qconfig
        if qconfig is None:
            self.qconfig = QConfig(act_type='int8', weight_type='int8')
        self.weight_observer=weight_observer
        self.activation_observer=activation_observer
        self.gradient_observer=gradient_observer
        self.mode = mode
        self.weight_converted = False
        self.quantize_gradient=False

    def forward(self, x):
        if self.mode == 'float':
            return super(FixedConv2d, self).forward(x)

        if not self.weight_converted:
            self.weight_scale = 2**7#self.weight_observer.observe(self.weight, self.qconfig.weight_type)
            self.weight.data = convert_to_8bit(self.weight.data, self.weight_scale, self.qconfig.weight_type)
            if self.bias is not None:
                self.bias.data = convert_to_8bit(self.bias.data, self.weight_scale, self.qconfig.weight_type)
            self.weight_converted = True

        if self.mode == 'observe':
            y = super(FixedConv2d, self).forward(x)

            if y.requires_grad:
                y.register_hook(add_gradient_observer_hook(self.gradient_observer))
        
        elif self.mode == 'quant':
            y = super(FixedConv2d, self).forward(x)
            #y.data = convert_to_8bit(y.data, self.scale, self.qtype)
            if y.requires_grad and self.quantize_gradient:
                y.register_hook(add_gradient_quantizer_hook(self.gradient_observer))
        return y

class FixedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        qconfig=None,
        mode='float',
        weight_observer=MinMaxObserver(),
        activation_observer=MinMaxObserver(),
        gradient_observer=MinMaxObserver()
    ): 

        super(FixedConvTranspose2d, self).__init__(in_channels,
                                                out_channels,
                                                kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                output_padding=output_padding,
                                                dilation=dilation,
                                                groups=groups,
                                                bias=bias,
                                                padding_mode=padding_mode)
        self.qconfig=qconfig
        if qconfig is None:
            self.qconfig = QConfig(act_type='int8', weight_type='int8')
        self.weight_observer=weight_observer
        self.activation_observer=activation_observer
        self.gradient_observer=gradient_observer
        self.mode = mode
        self.weight_converted = False
        self.quantize_gradient=False

    def forward(self, x):
        if self.mode == 'float':
            return super(FixedConvTranspose2d, self).forward(x)


        if not self.weight_converted:
            self.weight_scale = 2**7#self.weight_observer.observe(self.weight, self.qconfig.weight_type)
            self.weight.data = convert_to_8bit(self.weight.data, self.weight_scale, self.qconfig.weight_type)
            if self.bias is not None:
                self.bias.data = convert_to_8bit(self.bias.data, self.weight_scale, self.qconfig.weight_type)
            self.weight_converted = True

        if self.mode == 'observe':
            y = super(FixedConvTranspose2d, self).forward(x)
            if y.requires_grad:
                y.register_hook(add_gradient_observer_hook(self.gradient_observer))
        
        elif self.mode == 'quant':
            y = super(FixedConvTranspose2d, self).forward(x)
            if y.requires_grad and self.quantize_gradient:
                y.register_hook(add_gradient_quantizer_hook(self.gradient_observer))
        return y


class FixedAdd(nn.Module):
    def __init__(self,
                 qconfig=None, 
                 activation_observer=MinMaxObserver(),
                 gradient_observer=MinMaxObserver(),
                 mode='float'):

        super(FixedAdd, self).__init__()
        self.activation_observer=activation_observer
        self.gradient_observer=gradient_observer
        self.mode = mode
        self.qconfig=qconfig
        self.quantize_gradient=False
        if qconfig is None:
            self.qconfig = QConfig(act_type='int8')

    def forward(self, x1, x2):
        if self.mode == 'float':
            return x1+x2

        if self.mode == 'observe':
            y = x1 + x2
            self.scale = self.activation_observer.observe(y, self.qconfig.act_type)
            self.qtype = self.qconfig.act_type
            if y.requires_grad:
                y.register_hook(add_gradient_observer_hook(self.gradient_observer))

        elif self.mode == 'quant':
            y = x1 + x2
            y.data = convert_to_8bit(y.data, self.scale, self.qtype)
            if y.requires_grad and self.quantize_gradient:
                y.register_hook(add_gradient_quantizer_hook(self.gradient_observer))
        return y

    def add(self, x1, x2):
        return self.forward(x1, x2)


class FixedInputQuantizer(nn.Module):
    def __init__(self,
                 qconfig=None, 
                 activation_observer=MinMaxObserver(),
                 gradient_observer=MinMaxObserver(),
                 quantize_gradient=False,
                 show_act_quant=False,
                 mode='float'):

        super(FixedInputQuantizer, self).__init__()
        self.activation_observer=activation_observer
        self.gradient_observer=gradient_observer
        self.mode = mode
        self.quantize_gradient = quantize_gradient
        self.qconfig = qconfig
        if qconfig is None:
            self.qconfig = QConfig(act_type='int8')
        self.scale = torch.tensor([2**7])
        self.qtype = 'int8'
        self.show_act_quant = show_act_quant

    def forward(self, x):
        if self.mode == 'float':
            return x

        if self.mode == 'observe':
            y = x
            
            if y.requires_grad:
                y.register_hook(add_gradient_observer_hook(self.gradient_observer))

        elif self.mode == 'quant':
            if x.is_cuda:
                self.scale = self.scale.cuda()
            y = x
            y.data = convert_to_8bit(y.data, self.scale, self.qtype)
            if self.show_act_quant:
                with torch.no_grad():
                    self.output_value = y.data
            if y.requires_grad and self.quantize_gradient:
                y.register_hook(add_gradient_quantizer_hook(self.gradient_observer))

        return y





class FixedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, 
                 qconfig=None, 
                 weight_observer=MinMaxObserver(), 
                 activation_observer=MinMaxObserver(),
                 gradient_observer=MinMaxObserver(),
                 mode='float',
                 **kwargs):
        super(FixedBatchNorm2d, self).__init__(num_features, **kwargs)

        self.weight_observer=weight_observer
        self.activation_observer=activation_observer
        self.gradient_observer=gradient_observer
        self.weight_converted=False
        self.mode = mode
        self.quantize_gradient=False
        self.qconfig=qconfig
        if qconfig is None:
            self.qconfig = QConfig(act_type='int8', weight_type='int8')

    def call(self, x):
        if self.mode == 'float':
            return super(FixedBatchNorm2d, self).forward(x)
        if not self.weight_converted:
            self.weight_scale = self.weight_observer.observe(self.weight, self.qconfig.weight_type)
            self.weight_scale = self.weight_observer.observe(self.bias, self.qconfig.weight_type)
            self.weight_scale = self.weight_observer.observe(self.running_mean, self.qconfig.weight_type)
            self.weight_scale = self.weight_observer.observe(self.running_var, self.qconfig.weight_type)

            self.weight.data = convert_to_8bit(self.weight.data, self.weight_scale, self.qconfig.weight_type)
            self.bias.data = convert_to_8bit(self.bias.data, self.weight_scale, self.qconfig.weight_type)
            self.running_var.data = convert_to_8bit(self.running_mean.data, self.weight_scale, self.qconfig.weight_type)
            self.running_var.data = convert_to_8bit(self.running_var.data, self.weight_scale, self.qconfig.weight_type)

            self.weight_converted = True

        if self.mode == 'observe':
            y = super(FixedBatchNorm2d, self).forward(x)
            self.scale = self.activation_observer.observe(y, self.qconfig.act_type)
            self.qtype = self.qconfig.act_type

            if y.requires_grad:
                y.register_hook(add_gradient_observer_hook(self.gradient_observer))
        
        elif self.mode == 'quant':
            y = super(FixedBatchNorm2d, self).forward(x)
            if y.requires_grad and self.quantize_gradient:
                y.register_hook(add_gradient_quantizer_hook(self.gradient_observer))

        return y

   


class FixedReLU(nn.ReLU):
    def __init__(self,
                 noisy_activation=False,
                 qconfig=None, 
                 activation_observer=MinMaxObserver(),
                 gradient_observer=MinMaxObserver(),
                 quantize_gradient=False,
                 mode='float',
                 show_act_quant=False):
        super(FixedReLU, self).__init__()

        self.scale = 1.
        self.activation_observer=activation_observer
        self.gradient_observer=gradient_observer
        self.mode = mode
        self.quantize_gradient = quantize_gradient
        self.qconfig=qconfig
        self.noisy_activation = noisy_activation
        if qconfig is None:
            self.qconfig = QConfig(act_type='uint8')

        self.show_act_quant = show_act_quant


    def forward(self, x):
        if self.mode == 'float':
            if self.training and self.noisy_activation:
                u = 2**(-4)*(torch.cuda.FloatTensor(x.size()).uniform_() - 0.5)
                if x.is_cuda:
                    u = u.cuda()
                x = x+u
            return x.clamp_(0., 255./64.)

        elif self.mode == 'observe':
            y = x.clamp_(0., 255./64.)
            self.scale = torch.tensor([2**6])
            if x.is_cuda:
                self.scale = self.scale.cuda()
            self.qtype = self.qconfig.act_type

            if y.requires_grad:
                y.register_hook(add_gradient_observer_hook(self.gradient_observer))

        elif self.mode == 'quant':
            y = x.clamp_(0., 255./64.)
            #if self.show_act_quant:
            #    print(self.__class__.__name__, y.size(), "Before Quant: ", y.view(-1).detach().clone().cpu().numpy()[:10])
            y.data = convert_to_8bit(y.data, self.scale, self.qtype)
            if self.show_act_quant:
            #    print(self.__class__.__name__, y.size(), "After Quant: ", y.view(-1).detach().clone().cpu().numpy()[:10])
                with torch.no_grad():
                    self.output_value = y.data
            if y.requires_grad and self.quantize_gradient:
                y.register_hook(add_gradient_quantizer_hook(self.gradient_observer))

        return y

class FixedTanh(nn.Tanh):
    def __init__(self,
                 qconfig=None, 
                 activation_observer=MinMaxObserver(),
                 gradient_observer=MinMaxObserver(),
                 mode='float',
                 show_act_quant=False):

        super(FixedTanh, self).__init__()
        self.activation_observer=activation_observer
        self.gradient_observer=gradient_observer
        self.mode = mode
        self.qconfig=qconfig
        self.quantize_gradient=False
        if qconfig is None:
            self.qconfig = QConfig(act_type='uint8')

        self.show_act_quant = show_act_quant

    def forward(self, x):
        if self.mode == 'float':
            return super(FixedTanh, self).forward(x)

        elif self.mode == 'observe':
            y = super(FixedTanh, self).forward(x)
            self.scale = torch.tensor([2**7])
            if x.is_cuda:
                self.scale = self.scale.cuda()
            self.qtype = 'int8'

            if y.requires_grad:
                y.register_hook(add_gradient_observer_hook(self.gradient_observer))

        elif self.mode == 'quant':
            y = super(FixedTanh, self).forward(x)
            #if self.show_act_quant:
            #    print(self.__class__.__name__, y.size(), "Before Quant: ", y.view(-1).detach().clone().cpu().numpy()[:10])
            y.data = convert_to_8bit(y.data, self.scale, self.qtype, N=8)
            if self.show_act_quant:
                #print(self.__class__.__name__, y.size(), "After Quant: ", y.view(-1).detach().clone().cpu().numpy()[:10])
                with torch.no_grad():
                    self.output_value = y.data
            if y.requires_grad:
                y.register_hook(add_gradient_quantizer_hook(self.gradient_observer))

        return y


class FixedSigmoid(nn.Sigmoid):
    def __init__(self,
                 qconfig=None, 
                 activation_observer=MinMaxObserver(),
                 gradient_observer=MinMaxObserver(),
                 mode='float',
                 show_act_quant=False):

        super(FixedSigmoid, self).__init__()
        self.activation_observer=activation_observer
        self.gradient_observer=gradient_observer
        self.mode = mode
        self.qconfig=qconfig
        self.quantize_gradient=False
        if qconfig is None:
            self.qconfig = QConfig(act_type='uint8')

        self.show_act_quant = show_act_quant

    def forward(self, x):
        if self.mode == 'float':
            return super(FixedSigmoid, self).forward(x)

        elif self.mode == 'observe':
            y = super(FixedSigmoid, self).forward(x)
            self.scale = torch.tensor([2**8])
            if x.is_cuda:
                self.scale = self.scale.cuda()
            self.qtype = 'uint8'

        elif self.mode == 'quant':
            y = super(FixedSigmoid, self).forward(x)
            #if self.show_act_quant:
            #    print(self.__class__.__name__, y.size(), "Before Quant: ", y.view(-1).detach().clone().cpu().numpy()[:10])
            y.data = convert_to_8bit(y.data, self.scale, self.qtype, N=8)
            if self.show_act_quant:
                with torch.no_grad():
                    self.output_value = y.data
                #print(self.__class__.__name__, y.size(), "After Quant: ", y.view(-1).detach().clone().cpu().numpy()[:10])

        return y

