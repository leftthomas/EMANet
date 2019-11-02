import math
from typing import Dict

import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.meta_arch.semantic_seg import SEM_SEG_HEADS_REGISTRY
from torch import nn
from torch.nn import functional as F


class EMAUnit(nn.Module):
    """The Expectation-Maximization Attention Unit (EMA Unit).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        iteration_num (int): The iteration number for EM.
    """

    def __init__(self, c, k, iteration_num=3, em_mom=0.9, norm="BN"):
        super(EMAUnit, self).__init__()
        self.iteration_num = iteration_num

        mu = torch.Tensor(1, c, k)
        # init with Kaiming Norm
        mu.normal_(0, math.sqrt(2. / k))
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)
        self.em_mom = em_mom

        self.conv1 = Conv2d(c, c, kernel_size=1)
        self.conv2 = Conv2d(c, c, kernel_size=1, bias=False, norm=get_norm(norm, c))
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        # the EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # b * c * k

        for i in range(self.iteration_num):
            x_t = x.permute(0, 2, 1)  # b * n * c
            z = torch.bmm(x_t, mu)  # b * n * k
            z = F.softmax(z, dim=2)  # b * n * k
            z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
            mu = torch.bmm(x.detach(), z_.detach())  # b * c * k
            mu = self._l2norm(mu, dim=1)

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        if self.training:
            mu = mu.mean(dim=0, keepdim=True)
            self.mu *= self.em_mom
            self.mu += mu * (1 - self.em_mom)

        return x

    def _l2norm(self, inp, dim):
        """Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the sub-tensors.
        Returns:
            (tensor) The normalized tensor.
        """
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


@SEM_SEG_HEADS_REGISTRY.register()
class EMAHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        iteration_num = cfg.MODEL.SEM_SEG_HEAD.ITERATION_NUM
        em_mom = cfg.MODEL.EMA.EM_MOM
        # fmt: on

        self.reduced_conv = Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, 512),
                                   activation=F.relu)
        self.emau = EMAUnit(512, 64, iteration_num, em_mom, norm)
        self.predictor = nn.Sequential(
            Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, 256),
                   activation=F.relu), nn.Dropout2d(p=0.1), Conv2d(256, num_classes, kernel_size=1))
        weight_init.c2_msra_fill(self.reduced_conv)
        for module in self.predictor:
            if isinstance(module, Conv2d):
                weight_init.c2_msra_fill(module)

    def forward(self, features, size, targets=None):
        x = self.reduced_conv(features['res5'])
        x = self.emau(x)
        x = self.predictor(x)

        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)

        if self.training:
            losses = {"loss_sem_seg": (F.cross_entropy(x, targets, reduction="mean", ignore_index=self.ignore_value))}
            return [], losses
        else:
            return x, {}
