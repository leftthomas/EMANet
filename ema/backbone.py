import fvcore.nn.weight_init as weight_init
import numpy as np
import torch.nn.functional as F
from detectron2.layers import (
    Conv2d,
    FrozenBatchNorm2d,
    ShapeSpec,
    get_norm,
)
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from torch import nn


class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


class ResNetBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class BottleneckBlock(ResNetBlockBase):
    def __init__(self, inplanes, planes, stride=1, norm="BN", dilation=1):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN", "SyncBN"}).
        """
        super().__init__(inplanes, planes * 4, stride)

        if stride != 1 or inplanes != planes * 4:
            self.shortcut = Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False,
                                   norm=get_norm(norm, planes * 4))
        else:
            self.shortcut = None

        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False, norm=get_norm(norm, planes))

        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False,
                            dilation=dilation, norm=get_norm(norm, planes))

        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False, norm=get_norm(norm, planes * 4))

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


def make_stage(inplanes, planes, blocks, stride=1, dilation=1, norm="BN", grids=None):
    layers = []
    if grids is None:
        grids = [1] * blocks

    if dilation == 1 or dilation == 2:
        layers.append(BottleneckBlock(inplanes, planes, stride, norm=norm, dilation=1))
    elif dilation == 4:
        layers.append(BottleneckBlock(inplanes, planes, stride, norm=norm, dilation=2))
    else:
        raise RuntimeError('=> unknown dilation size: {}'.format(dilation))

    for i in range(1, blocks):
        layers.append(BottleneckBlock(planes * 4, planes, norm=norm, dilation=dilation * grids[i]))

    return layers


class DilatedResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(DilatedResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)
            nn.init.normal_(self.linear.weight, stddev=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_dilated_resnet_backbone(cfg, input_shape):
    """
    Create a Dilated ResNet instance from config.
    Returns:
        DilatedResNet: a :class:`DilatedResNet` instance.
    """
    # fmt: off
    norm = cfg.MODEL.DILATED_RESNET.NORM
    depth = cfg.MODEL.DILATED_RESNET.DEPTH
    stride = cfg.MODEL.DILATED_RESNET.STRIDE
    # fmt: on

    stem = BasicStem(in_channels=input_shape.channels, out_channels=128, norm=norm)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []
    layer1 = make_stage(128, 64, num_blocks_per_stage[0], norm=norm)
    layer2 = make_stage(64 * 4, 128, num_blocks_per_stage[1], stride=2, norm=norm)

    if stride == 16:
        layer3 = make_stage(128 * 4, 256, num_blocks_per_stage[2], stride=2, norm=norm)
        layer4 = make_stage(256 * 4, 512, num_blocks_per_stage[3], stride=1, norm=norm, dilation=2, grids=[1, 2, 4])
    elif stride == 8:
        layer3 = make_stage(128 * 4, 256, num_blocks_per_stage[2], stride=1, norm=norm, dilation=2)
        layer4 = make_stage(256 * 4, 512, num_blocks_per_stage[3], stride=1, norm=norm, dilation=4, grids=[1, 2, 4])
    else:
        raise RuntimeError('=> unknown stride size: {}'.format(stride))
    stages.append(layer1)
    stages.append(layer2)
    stages.append(layer3)
    stages.append(layer4)
    return DilatedResNet(stem, stages)
