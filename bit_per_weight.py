import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import numpy as np
from models.fixed_point_modules import convert_to_8bit
#from dahuffman import HuffmanCodec
from models.fixed_point_modules import FixedReLU, FixedAdd, FixedInputQuantizer, FixedTanh, FixedSigmoid, FixedConv2d, FixedConvTranspose2d
import torch.nn.utils.prune as prune
from test_gen import calc_bpw
from HuffmanCoding_weights import HuffmanCoding


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


def run_length_encoding_sim(weight, bit=5):
    weight = weight.reshape(-1)
    count = -1
    vals = []
    indices = []

    for i in range(len(weight)):
        w = weight[i]
        if w != 0:
            vals.append(w)
            indices.append(count+1)
            count = 0
        else:
            count += 1
            if count >= 2**bit:
                vals.append(0)
                indices.append(count)
                count = 0

    return vals, indices



    



def quantize(x, scale, dtype, N=8):
    with torch.no_grad():
        y = (x * scale).int()
        if dtype == 'uint8':
            y = torch.clamp(y, 0, 2**N-1)
            return y
        elif dtype == 'int8':
            y = torch.clamp(y, -2**(N-1), 2**(N-1)-1)
            return y
        else:
            raise NotImplementedError()




def pad_weight(weight):
    (oc, ic, krow, kcol) = weight.shape
    if oc < 8:
        weight = np.concatenate([weight, np.zeros((8-oc, ic, krow, kcol))], axis=0)
    (oc, ic, krow, kcol) = weight.shape
    if ic < 8:
        weight = np.concatenate([weight, np.zeros((oc, 8-ic, krow, kcol))], axis=1)
    return weight



def get_bpw(model, hw_mode=True, encoder_only=False):
    total_length = 0
    total_weights = 0
    total_sparsity = 0
    #total_params = list(model.netE.parameters()) + list(model.netDecoder.parameters())
        


    modules = model.netE.named_modules() if encoder_only else model.named_modules()


    for name, module in modules:
        if isinstance(module, FixedConvTranspose2d) or isinstance(module, FixedConv2d):

            p = module.weight

            p_ = p.clone()
            num_weight = len(p.reshape(-1))

            bias = module.bias
            length = len(bias)*8
            num_weight += len(bias)


            p = quantize(p, 2**7, 'int8')
            p = p.cpu().detach().numpy()


            if hw_mode:
                
                p = pad_weight(p)
                weight_coded, index_coded, wtable, itable = calc_bpw(p)

                wtable = np.array(wtable).reshape(-1).tolist()
                itable = np.array(itable).reshape(-1).tolist()
                

            else:
                vals, indices = run_length_encoding_sim(p)

                val_tree = HuffmanCoding(vals)
                idx_tree = HuffmanCoding(indices)

                weight_coded = val_tree.compress()
                index_coded = idx_tree.compress()

                wtable = val_tree.get_code_table()
                itable = idx_tree.get_code_table()

                wtable = np.array(wtable).reshape(-1).tolist()
                itable = np.array(itable).reshape(-1).tolist()

            
            wc = ''
            ic = ''
            wt = ''
            it = ''

            for w in weight_coded:
                wc += w
            for i in index_coded:
                ic += i
            for w in wtable:
                wt += w
            for i in itable:
                it += i

            length +=  len(ic)  + len(wc) + len(wt) + len(it) 

            sparsity = torch.sum(module.weight == 0).item() + torch.sum(module.bias == 0).item()
            total_length += length
            total_weights += num_weight
            total_sparsity +=  sparsity
            print("Sparsity: {:.2f}, BPW: {:.2f}".format(sparsity / num_weight * 100, length/num_weight) )




    print('bpp:',total_length / total_weights)
    print('Total Size (Mb):', total_length / 1024 / 1024)
    print('Total Sparsity:', total_sparsity / total_weights)



def main():
    opt = TestOptions().parse(save=False)
    opt.model = 'Bpgan_GAN_Q'

    model = create_model(opt)
    model.eval()
    get_bpw(model, True, True)

if __name__ == "__main__":
    main()
