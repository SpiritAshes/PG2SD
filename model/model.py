import torch
import torch.nn as nn
import torch.nn.functional as F


class MergedNet(nn.Module):
    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False, dim=128, mchan=4, relu22=False):
        super().__init__()
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])
        self.out_dim = dim

        # 构建网络层
        self._add_conv(8 * mchan)
        self._add_conv(8 * mchan)
        self._add_conv(16 * mchan, stride=2)
        self._add_conv(16 * mchan)
        self._add_conv(32 * mchan, stride=2)
        self._add_conv(32 * mchan)
        # 替换最后的 8x8 卷积为 3 个 2x2 卷积
        self._add_conv(32 * mchan, k=2, stride=2, relu=relu22)
        self._add_conv(32 * mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)

        # 可靠性和重复性分类器
        self.clf = nn.Conv2d(self.out_dim, 1, kernel_size=1)
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:, 1:2]

    def sigmoid(self, ux):
        return torch.sigmoid(ux)

    def normalize(self, x, ureliability=None, urepeatability=None):
        if ureliability is None:
            ureliability = torch.zeros_like(x)
        if urepeatability is None:
            urepeatability = torch.zeros_like(x)
        descriptors = F.normalize(x, p=2, dim=1)
        repeatability = self.softmax(urepeatability)
        reliability = self.sigmoid(ureliability)
        return descriptors, repeatability, reliability

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True, k_pool=1, pool_type='max'):
        d = self.dilation * dilation
        if self.dilated:
            conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=stride)
        self.ops.append(nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params))
        if bn and self.bn:
            self.ops.append(self._make_bn(outd))
        if relu:
            self.ops.append(nn.ReLU(inplace=True))
        self.curchan = outd

        if k_pool > 1:
            if pool_type == 'avg':
                self.ops.append(torch.nn.AvgPool2d(kernel_size=k_pool))
            elif pool_type == 'max':
                self.ops.append(torch.nn.MaxPool2d(kernel_size=k_pool))
            else:
                print(f"Error, unknown pooling type {pool_type}...")

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # 计算置信度图
        ureliability = self.clf(x)
        urepeatability = self.sal(x)
        return self.normalize(x, ureliability, urepeatability)

    def forward_list(self, imgs):
        descriptors_list = []
        repeatability_list = []
        reliability_list = []

        for img in imgs:
            descriptors, repeatability, reliability = self.forward_one(img)
            descriptors_list.append(descriptors)
            repeatability_list.append(repeatability)
            reliability_list.append(reliability)

        return descriptors_list, repeatability_list, reliability_list
    
    def forward(self, imgs):

        if isinstance(imgs, (list, tuple)):
            return self.forward_list(imgs)
        else:
            return self.forward_one(imgs)