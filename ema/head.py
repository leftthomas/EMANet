import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.meta_arch.semantic_seg import SEM_SEG_HEADS_REGISTRY
from torch import nn
from torch.nn import functional as F
from typing import Dict


@SEM_SEG_HEADS_REGISTRY.register()
class DANetHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        # fmt: on

        self.scale_pam_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_pam_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature + '_pam', self.scale_pam_heads[-1])

        self.scale_cam_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_cam_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature + '_cam', self.scale_cam_heads[-1])

        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                pam = self.scale_pam_heads[i](features[f])
                cam = self.scale_cam_heads[i](features[f])
            else:
                pam = pam + self.scale_pam_heads[i](features[f])
                cam = cam + self.scale_cam_heads[i](features[f])

        b, c, h, w = pam.size()
        B_T = pam.view(b, c, -1)
        B = B_T.transpose(-1, -2).contiguous()
        pam_weight = F.softmax(torch.matmul(B, B_T), dim=-1).view(b, 1, h * w, h * w)
        weighted_pam = torch.matmul(pam_weight, pam.view(b, c, h * w, 1)).view(b, c, h, w)
        sum_pam = pam + weighted_pam

        b, c, h, w = cam.size()
        A = cam.view(b, c, -1)
        A_T = A.transpose(-1, -2).contiguous()
        cam_weight = F.softmax(torch.matmul(A, A_T), dim=-1).view(b, 1, c, c)
        weighted_cam = torch.matmul(cam_weight, cam.view(b, c, h * w).transpose(-1, -2).contiguous()
                                    .view(b, h * w, c, 1)).view(b, h * w, c).transpose(-1, -2).contiguous() \
            .view(b, c, h, w)
        sum_cam = cam + weighted_cam

        x = self.predictor(sum_pam + sum_cam)
        x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        if self.training:
            losses = {}
            losses["loss_sem_seg"] = (F.cross_entropy(x, targets, reduction="mean", ignore_index=self.ignore_value))
            return [], losses
        else:
            return x, {}
