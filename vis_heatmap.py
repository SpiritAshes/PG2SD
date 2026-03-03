import pdb
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch.nn as nn
import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as pl; pl.ion()
from scipy.ndimage import uniform_filter

from utils.dataloader import norm_RGB
from model.model import MergedNet

smooth = lambda arr: uniform_filter(arr, 3)

def transparent(img, alpha, cmap, **kw):
    from matplotlib.colors import Normalize
    colored_img = cmap(Normalize(clip=True,**kw)(img))
    colored_img[:,:,-1] = alpha
    return colored_img

class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]
        # y_coords, x_coords = torch.where(maxima.squeeze())  # 先 squeeze 去掉 batch 和 channel 维度，得到 (H,W)
        # return torch.stack([y_coords, x_coords])
    
if __name__ == '__main__':
    # 加载配置文件
    with open('./config/test.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # 设置 GPU
    if config['gpu'][0] != -1:
        device = torch.device(f"cuda:{config['gpu'][0]}")
    else:
        device = torch.device("cpu")

    # 加载模型
    model = MergedNet().to(device)
    model.eval()

    if len(config['gpu']) > 1:
        model = torch.nn.DataParallel(model, device_ids=config['gpu'])
        orig_net = model.module
    else:
        orig_net = model

    if config['pretrained']:
        checkpoint = torch.load(config['pretrained_path'])
        model.load_state_dict(checkpoint['state_dict'])
        
        # 打印模型当前 state_dict 的所有键
        print("Model state_dict keys:")
        for key in model.state_dict().keys():
            print(key)


    # 加载图像
    img = Image.open(config['img']).convert('RGB')
    if config['resize']: img.thumbnail((config['resize'], config['resize']))
    img = np.asarray(img).copy()
        
    # 初始化非极大值抑制器
    detector = NonMaxSuppression(
        rel_thr=config['reliability_thr'], 
        rep_thr=config['repeatability_thr'])

    # 计算特征
    with torch.no_grad():
        print(">> computing features...")
        descriptors_list, repeatability_list, reliability_list = model(imgs=[norm_RGB(img).unsqueeze(0).to(device)])
        rela = reliability_list
        repe = repeatability_list
        kpts = detector(reliability_list, repeatability_list).T[:,[1,0]]
        kpts = kpts[repe[0][0,0][kpts[:,1],kpts[:,0]].argsort()[-config['max_kpts']:]]

    # 可视化
    fig = pl.figure("viz")
    kw = dict(cmap=pl.cm.RdYlGn, vmax=1)
    crop = (slice(config['border'], -config['border'] or 1),)*2
    

    ax1 = pl.subplot(131)
    pl.imshow(img[crop], cmap=pl.cm.gray)
    pl.xticks(()); pl.yticks(())

    x, y = kpts[:,0:2].cpu().numpy().T - config['border']
    pl.plot(x, y, '+', c=(0,1,0), ms=10, scalex=0, scaley=0)

    pl.subplot(132)
    pl.imshow(img[crop], cmap=pl.cm.gray)
    pl.xticks(()); pl.yticks(())
    c = repe[0][0,0].cpu().numpy()
    pl.imshow(transparent(smooth(c)[crop], 0.5, vmin=0, **kw))

    ax1 = pl.subplot(133)
    pl.imshow(img[crop], cmap=pl.cm.gray)
    pl.xticks(()); pl.yticks(())
    rela = rela[0][0,0].cpu().numpy()
    pl.imshow(transparent(rela[crop], 0.5, vmin=0.9, **kw))

    pl.gcf().set_size_inches(9, 2.73)
    pl.subplots_adjust(0.01, 0.01, 0.99, 0.99, hspace=0.1)

    pl.savefig(config['out'])
