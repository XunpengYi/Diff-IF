import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix, l_eps, l_max, l_grad, l_ssim, l_x0 = self.netG(self.data)
        #b, c, h, w = self.data['vis'].shape
        l_pix.backward()
        self.optG.step()

        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_eps'] = l_eps.item()
        self.log_dict['l_max'] = l_max.item()
        self.log_dict['l_grad'] = l_grad.item()
        self.log_dict['l_ssim'] = l_ssim.item()
        self.log_dict['l_x0'] = l_x0.item()

    def test(self, continous = False):
        self.netG.eval()
        self.input_data = self.data
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR, self.x0, self.denoise_x0 = self.netG.module.super_resolution(self.input_data, continous)
            else:
                self.SR, self.x0, self.denoise_x0 = self.netG.super_resolution(self.input_data, continous)
        self.netG.train()

    def test_ddim(self, continous = False):
        self.netG.eval()
        self.input_data = self.data
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR, self.x0, self.denoise_x0 = self.netG.module.super_resolution_ddim(self.input_data, continous)
            else:
                self.SR, self.x0, self.denoise_x0 = self.netG.super_resolution_ddim(self.input_data, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, sample=False):
        out_dict = OrderedDict()

        out_dict['DDPM'] = self.SR.unsqueeze(0).detach().float().cpu()
        out_dict['x0'] = self.x0.unsqueeze(0).detach().float().cpu()
        out_dict['denoise_x0'] = self.denoise_x0.unsqueeze(0).detach().float().cpu()
        out_dict['vis'] = self.data['vis'].detach().float().cpu()
        out_dict['ir'] = self.data['ir'].detach().float().cpu()
        out_dict['img_full'] = self.data['img_full'].detach().float().cpu()
        return out_dict

    def get_val_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['DDPM'] = self.SR.detach().float().cpu()
        out_dict['x0'] = self.x0.unsqueeze(0).detach().float().cpu()
        out_dict['denoise_x0'] = self.denoise_x0.unsqueeze(0).detach().float().cpu()
        out_dict['vis'] = self.data['vis'].detach().float().cpu()
        out_dict['ir'] = self.data['ir'].detach().float().cpu()
        out_dict['img_full'] = self.data['img_full'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step, best_psnr_flag = False):
        if best_psnr_flag == True:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_gen.pth')
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_opt.pth')
        else:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'Finall_gen.pth')
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'Finall_opt.pth')
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        #load_opt = False # not need opt
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path), strict=False)
            """if load_opt == True:
                opt_path = '{}_opt.pth'.format(load_path)
                if self.opt['phase'] == 'train':
                    opt = torch.load(opt_path)
                    self.optG.load_state_dict(opt['optimizer'])
                    self.begin_step = opt['iter']
                    self.begin_epoch = opt['epoch']"""
