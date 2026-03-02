import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class Sampler(nn.Module):
    """ Similar to NghSampler, but doesnt warp the 2nd image.
    Distance to GT =>  0 ... pos_d ... neg_d ... ngh
    Pixel label    =>  + + + + + + 0 0 - - - - - - -
    
    Subsample on query side: if > 0, regular grid
                                < 0, random points 
    In both cases, the number of query points is = W*H/subq**2
    """
    def __init__(self, ngh, subq=-8, subd=1, pos_d=2, neg_d=4, border=16,
                       maxpool_pos=True, subd_neg=-8):
        nn.Module.__init__(self)
        assert 0 <= pos_d < neg_d <= (ngh if ngh else 99)
        self.ngh = ngh
        self.pos_d = pos_d
        self.neg_d = neg_d
        assert subd <= ngh or ngh == 0
        assert subq != 0
        self.sub_q = subq
        self.sub_d = subd
        self.sub_d_neg = subd_neg
        if border is None: border = ngh
        assert border >= ngh, 'border has to be larger than ngh'
        self.border = border
        self.maxpool_pos = maxpool_pos
        self.precompute_offsets()

    def precompute_offsets(self):
        pos_d2 = self.pos_d**2
        neg_d2 = self.neg_d**2
        rad2 = self.ngh**2
        rad = (self.ngh//self.sub_d) * self.ngh # make an integer multiple
        pos = []
        neg = []
        for j in range(-rad, rad+1, self.sub_d):
          for i in range(-rad, rad+1, self.sub_d):
            d2 = i*i + j*j
            if d2 <= pos_d2:
                pos.append( (i,j) )
            elif neg_d2 <= d2 <= rad2: 
                neg.append( (i,j) )

        self.register_buffer('pos_offsets', torch.LongTensor(pos).view(-1,2).t())
        self.register_buffer('neg_offsets', torch.LongTensor(neg).view(-1,2).t())

    def gen_grid(self, step, aflow):
        B, two, H, W = aflow.shape
        dev = aflow.device
        b1 = torch.arange(B, device=dev)
        if step > 0:
            # regular grid
            x1 = torch.arange(self.border, W-self.border, step, device=dev)
            y1 = torch.arange(self.border, H-self.border, step, device=dev)
            H1, W1 = len(y1), len(x1)
            x1 = x1[None,None,:].expand(B,H1,W1).reshape(-1)
            y1 = y1[None,:,None].expand(B,H1,W1).reshape(-1)
            b1 = b1[:,None,None].expand(B,H1,W1).reshape(-1)
            shape = (B, H1, W1)
        else:
            # randomly spread
            n = (H - 2*self.border) * (W - 2*self.border) // step**2
            x1 = torch.randint(self.border, W-self.border, (n,), device=dev)
            y1 = torch.randint(self.border, H-self.border, (n,), device=dev)
            x1 = x1[None,:].expand(B,n).reshape(-1)
            y1 = y1[None,:].expand(B,n).reshape(-1)
            b1 = b1[:,None].expand(B,n).reshape(-1)
            shape = (B, n)
        return b1, y1, x1, shape

    def forward(self, feats, confs, aflow, **kw):
        B, two, H, W = aflow.shape
        assert two == 2
        feat1, conf1 = feats[0], (confs[0] if confs else None)
        feat2, conf2 = feats[1], (confs[1] if confs else None)
        
        # positions in the first image
        b1, y1, x1, shape = self.gen_grid(self.sub_q, aflow)

        # sample features from first image
        feat1 = feat1[b1, :, y1, x1]
        qconf = conf1[b1, :, y1, x1].view(shape) if confs else None
        
        #sample GT from second image
        b2 = b1
        xy2 = (aflow[b1, :, y1, x1] + 0.5).long().t()
        mask = (0 <= xy2[0]) * (0 <= xy2[1]) * (xy2[0] < W) * (xy2[1] < H)
        mask = mask.view(shape)
        
        def clamp(xy):
            torch.clamp(xy[0], 0, W-1, out=xy[0])
            torch.clamp(xy[1], 0, H-1, out=xy[1])
            return xy
        
        # compute positive scores
        xy2p = clamp(xy2[:,None,:] + self.pos_offsets[:,:,None])
        pscores = (feat1[None,:,:] * feat2[b2, :, xy2p[1], xy2p[0]]).sum(dim=-1).t()
#        xy1p = clamp(torch.stack((x1,y1))[:,None,:] + self.pos_offsets[:,:,None])
#        grid = FullSampler._aflow_to_grid(aflow)
#        feat2p = F.grid_sample(feat2, grid, mode='bilinear', padding_mode='border')
#        pscores = (feat1[None,:,:] * feat2p[b1,:,xy1p[1], xy1p[0]]).sum(dim=-1).t()
        if self.maxpool_pos:
            pscores, pos = pscores.max(dim=1, keepdim=True)
            if confs: 
                sel = clamp(xy2 + self.pos_offsets[:,pos.view(-1)])
                qconf = (qconf + conf2[b2, :, sel[1], sel[0]].view(shape))/2
        
        # compute negative scores
        xy2n = clamp(xy2[:,None,:] + self.neg_offsets[:,:,None])
        nscores = (feat1[None,:,:] * feat2[b2, :, xy2n[1], xy2n[0]]).sum(dim=-1).t()

        if self.sub_d_neg:
            # add distractors from a grid
            b3, y3, x3, _ = self.gen_grid(self.sub_d_neg, aflow)
            distractors = feat2[b3, :, y3, x3]
            dscores = torch.matmul(feat1, distractors.t())
            del distractors
            
            # remove scores that corresponds to positives or nulls
            dis2 = (x3 - xy2[0][:,None])**2 + (y3 - xy2[1][:,None])**2
            dis2 += (b3 != b2[:,None]).long() * self.neg_d**2
            dscores[dis2 < self.neg_d**2] = 0
            
            scores = torch.cat((pscores, nscores, dscores), dim=1)
        else:
            # concat everything
            scores = torch.cat((pscores, nscores), dim=1)

        gt = scores.new_zeros(scores.shape, dtype=torch.uint8)
        gt[:, :pscores.shape[1]] = 1

        return scores, gt, mask, qconf


class APLoss(nn.Module):
    """ 可微的平均精度（AP）损失函数，通过量化实现。
        
        输入:
            x: (N, M)   预测值，范围在 [min, max]
            label: (N, M) 真实标签，取值为 {0, 1}
        
        返回:
            每个查询的 AP 值（对于每个 n in {1..N}）
            注意：通常需要最小化 1 - mean(AP)
    """
    def __init__(self, nq=25, min=0, max=1, euc=False):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100  # 检查量化级别数是否合法
        self.nq = nq  # 量化级别数
        self.min = min  # 输入值的最小范围
        self.max = max  # 输入值的最大范围
        self.euc = euc  # 是否使用欧几里得距离
        gap = max - min
        assert gap > 0  # 检查范围是否合法
        
        # 初始化量化器：一个不可学习的固定卷积层
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        a = (nq-1) / gap  # 斜率
        
        # 第一半部分：线性函数，通过 (min+x, 1) 和 (min+x+1/a, 0)，其中 x = {nq-1..0}*gap/(nq-1)
        q.weight.data[:nq] = -a  # 权重为负斜率
        q.bias.data[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1))  # 偏置为 1 + a*(min+x)
        
        # 第二半部分：线性函数，通过 (min+x, 1) 和 (min+x-1/a, 0)，其中 x = {nq-1..0}*gap/(nq-1)
        q.weight.data[nq:] = a  # 权重为正斜率
        q.bias.data[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min)  # 偏置为 1 - a*(min+x)
        
        # 第一个和最后一个线性函数是特殊的：水平直线
        q.weight.data[0] = q.weight.data[-1] = 0  # 斜率为 0
        q.bias.data[0] = q.bias.data[-1] = 1  # 偏置为 1

    def compute_AP(self, x, label):
        """ 计算平均精度（AP）
        
        输入:
            x: (N, M)   预测值
            label: (N, M) 真实标签
        
        返回:
            每个样本的 AP 值
        """
        N, M = x.shape
        if self.euc:  # 如果使用欧几里得距离，将输入值转换为距离
            x = 1 - torch.sqrt(2.001 - 2*x)

        # 对所有预测值进行量化
        q = self.quantizer(x.unsqueeze(1))  # 通过量化器处理，形状变为 (N, 2*nq, M)
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # 取两部分的最小值并截断为 0，形状为 (N, nq, M)

        nbs = q.sum(dim=-1)  # 每个量化级别的样本数，形状为 (N, nq)
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # 每个量化级别的正确样本数，形状为 (N, nq)
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # 计算精度，形状为 (N, nq)
        rec /= rec.sum(dim=-1).unsqueeze(1)  # 归一化召回率，形状为 (N, nq)

        ap = (prec * rec).sum(dim=-1)  # 计算每个样本的 AP，形状为 (N,)
        return ap

    def forward(self, x, label):
        """ 前向传播
        
        输入:
            x: (N, M)   预测值
            label: (N, M) 真实标签
        
        返回:
            每个样本的 AP 值
        """
        assert x.shape == label.shape  # 检查输入形状是否一致
        return self.compute_AP(x, label)  # 调用 compute_AP 方法计算 AP
    


class ReliabilityLoss(nn.Module):
    """
    合并 PixelAPLoss 和 ReliabilityLoss 的类。
    计算像素级 AP 损失，同时可训练像素级置信度。

    feat1:  (B, C, H, W)   从 img1 提取的像素级特征
    feat2:  (B, C, H, W)   从 img2 提取的像素级特征
    aflow:  (B, 2, H, W)   绝对流: aflow[...,y1,x1] = x2,y2
    """
    def __init__(self, nq=20, base=0.5):
        nn.Module.__init__(self)
        self.aploss = APLoss(nq, min=0, max=1, euc=False)
        self.sampler = Sampler(ngh=7, subq=-8, subd=1, pos_d=2, neg_d=4, border=16, subd_neg=-8, maxpool_pos=True)
        assert 0 <= base < 1
        self.base = base

    def forward(self, descriptors_list, reliability_list, aflow):
        # 子采样
        scores, gt, msk, qconf = self.sampler(descriptors_list, reliability_list, aflow)
        # scores, gt, msk, qconf = self.sampler(descriptors, False, aflow)

        # 计算像素级 AP
        n = qconf.numel()
        if n == 0:
            return 0
        scores, gt = scores.view(n, -1), gt.view(n, -1)
        ap = self.aploss(scores, gt).view(msk.shape)

        pixel_loss = 1 - ap * qconf - (1 - qconf) * self.base

        loss = pixel_loss[msk].mean()
        return loss

