import os
import torch
import numpy as np
import json
from argparse import ArgumentParser
from models.models import create_model
from options.test_options import TestOptions
from models.fixed_point_modules import FixedReLU, FixedAdd, FixedInputQuantizer, FixedTanh, FixedSigmoid, FixedConv2d, FixedConvTranspose2d
import torch.nn.utils.prune as prune
from data.data_loader import CreateDataLoader
import util.util as util
import imageio



def main():
    opt = TestOptions().parse(save=False)
    opt.model = 'Bpgan_GAN_Q'

    if not os.path.exists("./parsed_models/"):
        os.makedirs("./parsed_models/")

    filename = opt.name
    filename += "-quant.json" if opt.fixed_point else "-float.json"
    output_path = os.path.join("./parsed_models/", filename)


    model = create_model(opt)
    model.eval()
   
    custom_dict = {}
    num_layers = 0
    for name, module in  model.named_modules():
        if isinstance(module, FixedConvTranspose2d) or isinstance(module, FixedConv2d):
            num_layers += 1
            print(torch.sum(module.weight == 0).item()/float(module.weight.nelement()))
            custom_dict["{}".format(num_layers)] = {"shape": list(module.weight.size()),
                                                    "W": module.weight.detach().cpu().numpy().tolist(),
                                                    "b": module.bias.detach().cpu().numpy().tolist(),
                                                    "type": "Conv2d" if isinstance(module, FixedConv2d) else "ConvTranspose2d"
                                                    }
            
    custom_dict["num_layers"] = num_layers
    with open(output_path, "w") as outputfile:
        json.dump(custom_dict, outputfile)

    print("Model parsed!")



def chip_test_mode(audio_model=False):
    opt = TestOptions().parse(save=False)
    opt.model = 'Bpgan_GAN_Q'

    output_base_dir = "./parsed_models/{}/weights".format(opt.name)
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    model = create_model(opt)
    model.eval()
    
    encoder = model.netE
    output_dir = os.path.join(output_base_dir, 'Encoder')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    i = 0
    for name, param in encoder.named_parameters():
        filename = os.path.join(output_dir, "conv{:02d}.{}".format((i//2)+1, name.split(".")[-1]))
        print(filename)
        np.save(filename, param.detach().cpu().numpy())
        i+=1

    output_dir = os.path.join(output_base_dir, 'Q')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, 'center')
    np.save(filename, model.Q.center.detach().cpu().numpy())



    if audio_model:
        decoder = model.netDecoder
        output_dir = os.path.join(output_base_dir, 'Decoder')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        i = 0
        for name, param in decoder.named_parameters():
            filename = os.path.join(output_dir, "conv{:02d}.{}".format((i//2)+1, name.split(".")[-1]))
            print(filename)
            np.save(filename, param.detach().cpu().numpy())
            i+=1


            

def inference(audio_model):
    opt = TestOptions().parse(save=False)
    opt.model = 'Bpgan_GAN_Q'

    output_base_dir = "./parsed_models/{}/activations".format(opt.name)
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    opt.batchSize=1
    opt.nThreads=1
    opt.serial_batches=True
    opt.no_flip=True
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    imtype = np.uint16 if opt.image_bit_num == 16 else np.uint8

    model = create_model(opt)
    model.eval()

    for name, m in model.named_modules():
        if hasattr(m, 'mode'):
            m.mode='observe'
    for i, data in enumerate(dataset):
        inputs, _ = model.encode_input(data['label'], infer=True)
        z = model.netE(inputs)
        quantized = model.Q(z, "Hard")
    
        reconstructed = model.netDecoder(quantized)
        break

    for name, m in model.named_modules():
        if hasattr(m, 'mode'):
            m.mode='quant'
        if isinstance(m, FixedInputQuantizer) or isinstance(m, FixedReLU) or isinstance(m, FixedSigmoid) or isinstance(m, FixedTanh):
            m.show_act_quant = True


    for i, data in enumerate(dataset):
        inputs, _ = model.encode_input(data['label'], infer=True)
        z = model.netE(inputs)
        quantized = model.Q(z, "Hard")

        reconstructed = model.netDecoder(quantized)
    
        break

    j = 1
    output_dir = os.path.join(output_base_dir, 'Encoder')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for name, m in model.netE.named_modules():
        if isinstance(m, FixedInputQuantizer):
            filename = os.path.join(output_dir, "encoder_input_image")

        elif isinstance(m, FixedReLU):
            filename = os.path.join(output_dir, "conv{:02d}_relu4".format(j))
            j+=1
        elif isinstance(m, FixedSigmoid):
            filename = os.path.join(output_dir, "conv{:02d}_sigmoid".format(j))
        else:
            continue
        np.save(filename, m.output_value.detach().cpu().numpy())

    filename = os.path.join(output_dir, "quantized_latent_vector")
    np.save(filename, quantized.detach().cpu().numpy())

    input_image = util.tensor2im(inputs[0], imtype=imtype)
    filename = os.path.join(output_base_dir, "input_image.png")
    imageio.imwrite(filename, input_image)

    with open(os.path.join("./parsed_models/{}".format(opt.name), 'arguments.txt'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    
    if audio_model:
        j = 1
        output_dir = os.path.join(output_base_dir, 'Decoder')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for name, m in model.netE.named_modules():
            if isinstance(m, FixedReLU):
                filename = os.path.join(output_dir, "conv{:02d}_relu4".format(j))
                j+=1
            elif isinstance(m, FixedTanh):
                filename = os.path.join(output_dir, "conv{:02d}_tanh".format(j))
            else:
                continue
            np.save(filename, m.output_value.detach().cpu().numpy())


def decode(audio_model=False):

    opt = TestOptions().parse(save=False)

    name = "audio_compression" if audio_model else "image_compression"
    with open(os.path.join("./parsed_models/{}".format(name), 'arguments.txt'), 'r') as f:
        opt.__dict__ = json.load(f)

    imtype = np.uint16 if opt.image_bit_num == 16 else np.uint8
    opt.model = 'Bpgan_GAN_Q'

    output_base_dir = "./parsed_models/{}/activations".format(opt.name)
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    model = create_model(opt)
    model.eval()

    received = np.load(os.path.join(output_base_dir, "Encoder", "quantized_latent_vector.npy"))
    received = torch.tensor(received)
    if len(opt.gpu_ids) > 0:
        received = received.cuda()

    decoded = model.netDecoder(received)
    filename = os.path.join(output_base_dir, "decoded.png")

    decoded_image = util.tensor2im(decoded[0], imtype=imtype)
    imageio.imwrite(filename, decoded_image)

    return decoded_image


if __name__ == "__main__":
    #main()
    chip_test_mode(False)
    inference(False)
    decode(False)
    
