import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel
import networks as networks
import lr_scheduler
from base_model import BaseModel
from loss import GANLoss, QC_GradientPenaltyLoss
import wandb
from edge_loss import sobel
from torch.autograd import Variable
from linePromgram import H_Star_Solution



logger = logging.getLogger('base')


class F_Model(BaseModel):
    def __init__(self, opt):
        super(F_Model, self).__init__(opt)
        self.edge_enhance = opt['train']['edge_enhance']

        # if opt['dist']:
        #     self.rank = torch.distributed.get_rank()
        # else:
        self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netD = networks.define_D(opt).to(self.device)
        # if opt['dist']:
        #     self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        # else:
        #     self.netG = DataParallel(self.netG)
        self.netG = DataParallel(self.netG)
        self.netD = DataParallel(self.netD)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            train_opt = opt['train']
            #self.init_model() # Not use init is OK, since Pytorch has its owen init (by default)
            self.netG.train()
            self.netD.train()

            # loss
            if train_opt["pixel_weight"] > 0:
                l_pix_type = train_opt["pixel_criterion"]
                if l_pix_type == "l1":
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == "l2":
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(f"Loss type {l_pix_type} not recognized.")
                self.l_pix_w = train_opt["pixel_weight"]
            else:
                print("Remove pixel loss")
                self.cri_pix = None

            if train_opt["feature_weight"] > 0:
                l_fea_type = train_opt["feature_criterion"]
                if l_fea_type == "l1":
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == "l2":
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(f"Feature type {l_fea_type} not recognized.")
                self.l_fea_w = train_opt["feature_weight"]

            else:
                print("Remove feature loss.")
                self.cri_fea = None

            if self.cri_fea: # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                self.netF = DataParallel(self.netF)
                # self.netF = DistributedDataParallel(self.netF, device_ids=[torch.cuda.current_device()])

            self.cri_gan = GANLoss(train_opt["gan_type"], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            self.D_update_ratio = train_opt["D_update_ratio"] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt["D_init_iters"] else 0

            self.WGAN_QC_regul = QC_GradientPenaltyLoss()

             ## Code for edge loss
            if self.edge_enhance:
                self.l_edge_w = train_opt["edge_weight"]   
                if train_opt["edge_type"] == "sobel":
                    self.cril_edge = sobel
                # elif train_opt["edge_type"] == "canny":
                #     self.cril_edge = canny
                # elif train_opt['edge_type'] == "hednet":
                #     self.netEdge = HedNet().cuda()
                #     for p in self.netEdge.parameters():
                #         p.requires_grad = False
                #     self.cril_edge = self.netEdge
                else:
                    raise NotImplementedError(f"Edge type {train_opt['edge_type']} not recognized.")


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            #self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], momentum=0.9)
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                print('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def init_model(self, scale=0.1):
        # Common practise for initialization.
        for layer in self.netG.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
                layer.weight.data *= scale  # for residual block
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias.data, 0.0)


    def feed_data(self, data, LR_img, ker_map):
        # self.var_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data.to(self.device)  # GT
        self.var_L, self.ker = LR_img.to(self.device), ker_map.to(self.device)
        # self.real_ker = data['real_ker'].to(self.device)  # real kernel map
        #self.ker = data['ker'].to(self.device) # [Batch, 1, k]
        #m = self.ker.shape[0]
        #self.ker = self.ker.view(m, -1)

    # def optimize_parameters(self, step):
    #     self.optimizer_G.zero_grad()
    #     self.fake_H = self.netG(self.var_L, self.ker)
    #     l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
    #     l_pix.backward()
    #     self.optimizer_G.step()

    #     # set log
    #     self.log_dict['l_pix'] = l_pix.item()

    def optimize_parameters(self, step):
        # Generator
        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L, self.ker)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix: # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix 
            
            if self.cri_fea: # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            
            pred_g_fake = self.netD(self.fake_H)
            if self.opt['train']['gan_type'] == "gan":
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real = self.netD(self.var_ref).detach()
                l_g_gan = self.l_gan_w * ( 
                    self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            elif self.opt['train']['gan_type'] == "wgan-qc":
                pred_d_real = self.netD(self.var_ref).detach()
                l_g_gan = self.l_gan_w *pow(pred_d_real.mean() - pred_g_fake.mean(), 2)

            l_g_total += l_g_gan

            if self.edge_enhance:
                real_edge = self.cril_edge(self.var_H)
                fake_edge = self.cril_edge(self.fake_H)
                edge_diff = real_edge - fake_edge
                edge_squa = edge_diff * edge_diff
                l_g_edge = self.l_edge_w * edge_squa.mean()

                l_g_total += l_g_edge

            if step % 50 == 0:
                # logger.info(f"Generator Loss: {l_g_total} at step {step}")
                metric = {"Generator Loss": l_g_total,
                          "pixel_loss": l_g_pix,
                          "feature_loss": l_g_fea,
                          "wgan-qc": l_g_gan,
                          "edge_loss": l_g_edge}
                wandb.log(metric, step=step)
            l_g_total.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 5)
            self.optimizer_G.step()

        # Discriminator
            
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.var_ref)

        
        if self.opt['train']['gan_type'] == 'gan':
            pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif self.opt['train']['gan_type'] == 'ragan':
            pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2           
        #Revised by zezengli on Oct. 7, 2020 ,loss=wgan_qc+ gamma*regulation
        elif self.opt['train']['gan_type'] == 'wgan-qc':
            fake_H_detach = Variable(self.fake_H.detach(),requires_grad=True)
            pred_d_fake = self.netD(fake_H_detach)  # detach to avoid BP to G
                       
            fakeImg = self.fake_H.detach().cpu()
            trueImg = self.var_ref.detach().cpu()

            HStar_real, HStar_fake = H_Star_Solution(fakeImg, trueImg, self.opt['train']['WQC_KCoef']) 
            HStar_real_tensor = Variable(torch.FloatTensor(HStar_real),requires_grad=False).to(self.device)
            HStar_fake_tensor = Variable(torch.FloatTensor(HStar_fake),requires_grad=False).to(self.device)

            pred_HStar_real = [pred_d_real, HStar_real_tensor]
            pred_HStar_fake = [pred_d_fake, HStar_fake_tensor]    
            l_d_total = self.opt['train']['WQC_gamma']*self.WGAN_QC_regul(pred_d_fake, self.var_ref, fake_H_detach, self.opt['train']['WQC_KCoef'])
            
            l_d_real = self.cri_gan(pred_HStar_real, True)
            l_d_fake = self.cri_gan(pred_HStar_fake, False)
            l_d_total += (l_d_real + l_d_fake) / 2
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))    
        if step % 50 == 0:
                # logger.info(f"Discriminator Loss: {l_d_total} at step {step}")
                wandb.log({"Discriminator Loss": l_d_total}, step=step)
        l_d_total.backward()
        self.optimizer_D.step()

        self.d_total_loss=l_d_total.detach().cpu()
        self.g_total_loss=l_g_total.detach().cpu()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_SR = self.netG(self.var_L, self.ker)
        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_SR.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['ker'] = self.ker.detach()[0].float().cpu()
        out_dict['Batch_SR'] = self.fake_SR.detach().float().cpu() # Batch SR, for train
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
