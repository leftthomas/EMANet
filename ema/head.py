from typing import Dict

import fvcore.nn.weight_init as weight_init
import numpy as np
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.meta_arch.semantic_seg import SEM_SEG_HEADS_REGISTRY
from torch import nn
from torch.nn import functional as F


@SEM_SEG_HEADS_REGISTRY.register()
class EMANetHead(nn.Module):
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

        self.scale_heads = []
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
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])

        # TODO

        x = self.predictor(x)
        x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        if self.training:
            losses = {}
            losses["loss_sem_seg"] = (F.cross_entropy(x, targets, reduction="mean", ignore_index=self.ignore_value))
            return [], losses
        else:
            return x, {}
