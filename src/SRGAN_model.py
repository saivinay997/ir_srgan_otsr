import torch
import torch.nn as nn
import networks
from torch.nn.parallel import DistributedDataParallel
from loss import GANLoss, QC_GradientPenaltyLoss
from collections import OrderedDict
from torch.autograd import Variable
from linePromgram import H_Star_Solution
from base_model import BaseModel
import lr_scheduler

class SRGANModel(BaseModel):
    def __init__(self, opt, train=True):
        super(SRGANModel, self).__init__(opt)

        # Device agnostic code 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # define the networks
        self.netG = networks.define_G(opt).to(self.device)
        self.netD = networks.define_D(opt).to(self.device)
        # if self.device == "cuda":
        #     self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        #     self.netD = DistributedDataParallel(self.netD, device_ids=[torch.cuda.current_device()])
        if train:
            # set the G and D networks on train mode
            self.netG.train()
            self.netD.train()

        train_opt = opt["train"]

        if train:
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
                # self.netF = DistributedDataParallel(self.netF, device_ids=[torch.cuda.current_device()])

            self.cri_gan = GANLoss(train_opt["gan_type"], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            self.D_update_ratio = train_opt["D_update_ratio"] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt["D_init_iters"] else 0

            self.WGAN_QC_regul = QC_GradientPenaltyLoss()


            ## Remove edge loss as of now
            self.cril_edge = None

            wd_G = train_opt['weight_decay_G'] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for k, v in self.netG.named_parameters(): 
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        print(f"Params {k} will not optimize.")
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay = wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            
            wd_D = train_opt["weight_decay_D"] if train_opt["weight_decay_D"] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas = (train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         gamma=train_opt['lr_gamma']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        self.load()

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print("Loading Model for G", load_path_G)
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])
    
    def feed_data(self, hr_imgs, lr_imgs, need_GT=True):
        self.var_L = lr_imgs.to(self.device)
        if need_GT:
            self.var_H = hr_imgs.to(self.device)
            # input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = hr_imgs.to(self.device)
    
    def optimize_parameters(self, step):
        # Generator
        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L.detach())

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
        
        l_d_total.backward()
        self.optimizer_D.step()

        self.d_total_loss=l_d_total.detach().cpu()
        self.g_total_loss=l_g_total.detach().cpu()

    def save(self, iter_step):
        self.save_network(self.netG, "G", iter_step)
        self.save_network(self.netD, "D", iter_step)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["LQ"] = self.var_L.detach()[0].float().cpu()
        out_dict["SR"] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.var_H.detach()[0].float().cpu()
        return out_dict

        