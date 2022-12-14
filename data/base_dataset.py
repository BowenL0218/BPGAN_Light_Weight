import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.transforms import functional as F
class ToTensor(object):
    def __init__(self,bit_num=8):
        self.bit_num = bit_num
    def __call__(self, pic):
        if self.bit_num == 8:
            img = F.to_tensor(pic)
            return img
        else:
            img = torch.from_numpy(np.expand_dims(np.array(pic).astype(np.float32), 0))
            return img.float()/(2**self.bit_num)
    def __repr__(self):
        return self.__class__.__name__ + '()'
class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    if opt.resize_or_crop == 'vga':
        x = random.randint(0, np.maximum(0, new_w - 640))
        y = random.randint(0, np.maximum(0, new_h - 480))
    else:
        x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
        y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True,Color_Input="gray"):
    transform_list = []
    if opt.phase=='test':
        #transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))
        pass
    else :
        if 'random_resized_crop' in opt.resize_or_crop:
            osize = [opt.loadSize, opt.loadSize]
            transform_list.append(transforms.RandomResizedCrop(osize, scale=(0.05, 0.2), interpolation=method))
        else:
            if 'vga' in opt.resize_or_crop:
                transform_list.append(transforms.Lambda(lambda img: __crop_vga(img, params['crop_pos'])))
            if 'resize' in opt.resize_or_crop:
                osize = [opt.loadSize, opt.loadSize]
                transform_list.append(transforms.Scale(osize, method))
            elif 'scale_width' in opt.resize_or_crop:
                transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))

            if 'crop' in opt.resize_or_crop:
                transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))



            if opt.resize_or_crop == 'none':
                base = float(2 ** opt.n_downsample_global)
                if opt.netG == 'local':
                    base *= (2 ** opt.n_local_enhancers)
                transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

        if opt.isTrain and not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomRotation(90))
    if opt.resize_or_crop == "crop_and_scale":
        transform_list =[]
        transform_list.append(transforms.CenterCrop(opt.loadSize))
        transform_list.append(transforms.Resize(opt.fineSize))
    transform_list += [ToTensor(opt.image_bit_num)]


    if normalize:
        if Color_Input == "gray":
            transform_list += [transforms.Normalize([0.5],[0.5])]
        elif Color_Input == "RGB":
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + min(tw,ow), y1 + min(th,oh)))
    return img


def __crop_vga(img, pos):
    ow, oh = img.size
    x1, y1 = pos
    tw = 640
    th = 480
    if (ow >= tw and oh >= th):        
        return img.crop((x1, y1, x1 + min(tw,ow), y1 + min(th,oh)))
    else:
        return img.resize((640, 480), Image.BICUBIC)
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
