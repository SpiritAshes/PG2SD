import pdb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from  math import *


class PeakLoss(nn.Module):
    """ Try to make the repeatability locally peaky.

    Mechanism: we maximize, for each pixel, the difference between the local mean
               and the local max.
    """
    def __init__(self, N=16):
        nn.Module.__init__(self)
        assert N % 2 == 0, 'N must be pair'
        self.preproc = nn.AvgPool2d(3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(N+1, stride=1, padding=N//2)
        self.avgpool = nn.AvgPool2d(N+1, stride=1, padding=N//2)

    def diff_function(self, sali):
        sali = self.preproc(sali) # 剔除奇异点
        diff_peak = self.maxpool(sali) - self.avgpool(sali)
        peak_loss = 1 - diff_peak.mean()
        return peak_loss    # 这里可以考虑2倍的缩小，根据实际训练情况进行调整


    def forward(self, repeatability_list):
        sali1 = repeatability_list[0]
        sali2 = repeatability_list[1]
        return (self.diff_function(sali1) + self.diff_function(sali2)) /2