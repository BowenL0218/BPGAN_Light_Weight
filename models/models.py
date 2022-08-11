import torch

def create_model(opt):
    print(opt.model)
    if opt.model == 'Bpgan_GAN':
    	#from .Audio_GAN_model import Audio_GAN_Model
    	#model = Audio_GAN_Model()
        from .Bpgan_GAN_model import Bpgan_GAN_Model
        model = Bpgan_GAN_Model()
    elif opt.model == 'Bpgan_GAN_Q':
        #from .Audio_GAN_Q_model import Audio_GAN_Q_Model
        #model = Audio_GAN_Q_Model()
        from .Bpgan_GAN_Q_model import Bpgan_GAN_Q_Model
        model = Bpgan_GAN_Q_Model()
    elif opt.model == 'Bpgan_GAN_Q_Compressed':
        from .Bpgan_GAN_Q_Compressed import Bpgan_GAN_Q_Compressed
        model = Bpgan_GAN_Q_Compressed()
    elif opt.model == 'Bpgan_Q_Compressed':
        from .Bpgan_Q_Compressed import Bpgan_Q_Compressed
        model = Bpgan_Q_Compressed()
    elif opt.model == 'Dis_Bpgan_GAN':
        from .Bpgan_GAN_Distillation_model import Bpgan_GAN_DIS_Model
        model = Bpgan_GAN_DIS_Model()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.qint8:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
