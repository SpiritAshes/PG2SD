import pdb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from  math import *

def flow_to_grid(aflow):
    H, W = aflow.shape[2:]
    grid = aflow.permute(0,2,3,1).clone()
    grid[:,:,:,0] *= 2/(W-1)
    grid[:,:,:,1] *= 2/(H-1)
    grid -= 1
    grid[torch.isnan(grid)] = 9e9 # invalids
    return grid


class CosimLoss (nn.Module):
    """ Try to make the repeatability repeatable from one image to the other.
    """
    def __init__(self, N=16):
        nn.Module.__init__(self)
        self.patches = nn.Unfold(N, padding=0, stride=N//2)

    def extract_patches(self, sal):
        patches = self.patches(sal).transpose(1,2) # flatten
        patches = F.normalize(patches, p=2, dim=2) # norm
        return patches
        
    def forward(self, repeatability_list, aflow):
        B,two,H,W = aflow.shape
        assert two == 2

        # normalize
        sali1, sali2 = repeatability_list
        grid = flow_to_grid(aflow)
        sali2 = F.grid_sample(sali2, grid, mode='bilinear', padding_mode='border', align_corners=True)

        patches1 = self.extract_patches(sali1)
        patches2 = self.extract_patches(sali2)
        cosim = (patches1 * patches2).sum(dim=2)
        return 1 - cosim.mean()