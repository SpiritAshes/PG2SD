import torch
import torch.nn as nn
from  math import *


class PGLLoss (nn.Module):
    """ Try to make the repeatability locally peaky.

    Mechanism: we maximize, for each pixel, the difference between the local mean
               and the local max.
    """
    def __init__(self, N=16, epoch_all=30, control_factor=0.2):
        nn.Module.__init__(self)
        assert N % 2 == 0, 'N must be pair'
        self.preproc = nn.AvgPool2d(3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(N+1, stride=1, padding=N//2)
        self.avgpool = nn.AvgPool2d(N+1, stride=1, padding=N//2)
        self.epoch = 1
        self.epoch_all = epoch_all
        self.control_factor = control_factor
        
    def gradient_function(self, diff, factor):
        Confidence_value = torch.abs(diff / 2)
        parameter_a = 1 / (1 - exp(-10 * (1 - factor)))
        grad_adjust = 0.001 * Confidence_value
        parameter_b = (self.epoch - 1) / self.epoch_all
        loss = torch.pow(Confidence_value, 2)  + (parameter_a * Confidence_value - torch.pow(Confidence_value, 2)) / (1 + exp(-10 * (parameter_b - factor)))
        loss = 1 - (loss - grad_adjust).mean()
        return loss

    def diff_function(self, sali):
        sali = self.preproc(sali) 
        diff_peak = self.maxpool(sali) - self.avgpool(sali)
        grad_loss = self.gradient_function(diff_peak, self.control_factor)
        return 0.01 * grad_loss  # 这里可以考虑2倍的缩小，根据实际训练情况进行调整


    def forward(self, repeatability_list):
        sali1 = repeatability_list[0]
        sali2 = repeatability_list[1]
        return (self.diff_function(sali1) + self.diff_function(sali2)) /2