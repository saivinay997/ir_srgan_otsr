from RRDBNet_arch import RRDBNet
from discriminator_vgg_arch import Discriminator_VGG_128, VGGFeatureExtractor
import torch
import sftmd_arch



def define_G(opt):
    
    opt_net = opt["network_G"]
    which_model = opt_net['which_model_G']
    if which_model == "RRDBNet":
        netG = RRDBNet(in_nc=opt_net["in_nc"], out_nc=opt_net["out_nc"], nf=opt_net["nf"], nb=opt_net["nb"])
    elif which_model == 'Predictor':
        netG = sftmd_arch.Predictor(in_nc=opt_net['in_nc'], nf=opt_net['nf'], code_len=opt_net['code_length'])
    elif which_model == 'Corrector':
        netG = sftmd_arch.Corrector(in_nc=opt_net['in_nc'], nf=opt_net['nf'], code_len=opt_net['code_length'])
    elif which_model == 'SFTMD':
        netG = sftmd_arch.SFTMD(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'], input_para=opt_net['code_length'])
    return netG

def define_D(opt):

    opt_net = opt["network_D"]
    netD = Discriminator_VGG_128(in_nc=opt_net["in_nc"], nf=opt_net["nf"])

    return netD

def define_F(opt, use_bn=False):
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                               use_input_norm=True, device=device)
    netF.eval()
    return netF