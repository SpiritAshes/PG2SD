import torch
import torch.nn as nn
import torch.nn.functional as F
from model.reliability_loss import ReliabilityLoss
from model.cos_loss import CosimLoss
from model.peak_loss import PeakLoss
from model.PGL_loss import PGLLoss
from model.G2SDL_loss import Distillation_Loss


class MultiLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.relia_loss = ReliabilityLoss(nq=config['reliability_nq'], base=config['reliability_base'])
        self.cos_loss = CosimLoss(N=config['cosim_patch_size'])
        self.peak_loss = PeakLoss(N=config['peak_patch_size'])
        self.PGL_loss = PGLLoss(N=config['PGL_patch_size'], epoch_all=config['total_epochs'], control_factor=config['gradient_control_factor'])
        self.G2SDL_loss = Distillation_Loss(epoch_all=config['total_epochs'], temperature=config['temperature'], control_factor=config['gradient_control_factor'], beta=config['beta'])

        self.G2SDL_loss_weight = config['G2SDL_loss_weight']

    def set_epoch(self, epoch):
        self.PGL_loss.epoch = epoch
        self.G2SDL_loss.epoch = epoch

    def forward(self, descriptors_list, repeatability_list, reliability_list, aflow):
        relia_loss = self.relia_loss(descriptors_list, reliability_list, aflow)
        cos_loss = self.cos_loss(repeatability_list, aflow)
        peak_loss = self.peak_loss(repeatability_list)
        PGL_loss = self.PGL_loss(repeatability_list)
        G2SDL_loss = self.G2SDL_loss(descriptors_list, repeatability_list) * self.G2SDL_loss_weight
        total_loss = relia_loss + cos_loss + peak_loss + PGL_loss + G2SDL_loss
        return total_loss, [relia_loss, cos_loss, peak_loss, PGL_loss, G2SDL_loss]