import torch
import torch.nn as nn
import torch.nn.functional as F
from  math import *


class Distillation_Loss(nn.Module):
    def __init__(self, epoch_all=30, temperature=1.5, control_factor=0.2, beta=0.5):
        nn.Module.__init__(self)

        self.epoch = 1
        self.epoch_all = epoch_all

        self.base_temperature = temperature
        self.base_beta = beta
        self.control_factor = control_factor

        self.smooth_loss = nn.SmoothL1Loss()

    def compute_schedule(self):
        temperature = max(
            self.base_temperature - (
                0.5 * (self.epoch - 1)
                / ((self.control_factor + 0.4) * self.epoch_all)
            ),
            1.0
        )

        beta = self.base_beta * max(
            (
                ((self.control_factor + 0.4) * self.epoch_all)
                - self.epoch + 1
            )
            / ((self.control_factor + 0.4) * self.epoch_all),
            0.0
        )

        return temperature, beta

    def forward(self, reliability, repeatability):

        temperature, beta = self.compute_schedule()

        # desc1, desc2 = descriptors
        relia1, relia2 = reliability
        repeat1, repeat2 = repeatability
        map_1 = torch.cat([relia1, repeat1], dim=1)
        map_2 = torch.cat([relia2, repeat2], dim=1)

        soft_lab_1 = F.softmax(map_1 / temperature, dim=1)
        soft_lab_2 = F.softmax(map_2 / temperature, dim=1)
        loss_map_1 = self.smooth_loss(map_1, soft_lab_1)
        loss_map_2 = self.smooth_loss(map_2, soft_lab_2)

        soft_log_map_1 = F.log_softmax(map_1 / temperature, dim=1)
        soft_log_map_2 = F.log_softmax(map_2 / temperature, dim=1)
        loss_self_distillation_1 = -torch.mean(torch.sum(soft_lab_1 * soft_log_map_1, dim=1))
        loss_self_distillation_2 = -torch.mean(torch.sum(soft_lab_2 * soft_log_map_2, dim=1))
        loss = (beta * (loss_map_1 + loss_map_2) + (1 - beta) * (loss_self_distillation_1 + loss_self_distillation_2)) / 2

        return loss







