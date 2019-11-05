import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from torch import nn

from .arch import get_norm


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    Parameters
    ----------
    norm_layer (str or callable): a callable that takes the number of
        channels and return a `nn.Module`, or a pre-defined string
        (one of {"BN", "SyncBN"}).
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer='BN', bn_mom=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = get_norm(norm_layer, planes, bn_mom)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                            bias=False)
        self.bn2 = get_norm(norm_layer, planes, bn_mom)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = get_norm(norm_layer, planes * 4, bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DilatedResNet(Backbone):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.
    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    norm_layer (str or callable): a callable that takes the number of
        channels and return a `nn.Module`, or a pre-defined string
        (one of {"BN", "SyncBN"}).
    """

    def __init__(self, layers, stride=8, norm_layer='BN', bn_mom=0.1):
        self.inplanes = 128
        assert stride in [8, 16], RuntimeError("=> unknown stride size: {}".format(stride))
        super(DilatedResNet, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm(norm_layer, 64, bn_mom),
            nn.ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            get_norm(norm_layer, 64, bn_mom),
            nn.ReLU(inplace=True),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.bn1 = get_norm(norm_layer, self.inplanes, bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], norm_layer=norm_layer, bn_mom=bn_mom)
        self.layer2 = self._make_layer(128, layers[1], stride=2, norm_layer=norm_layer, bn_mom=bn_mom)
        if stride == 8:
            self.layer3 = self._make_layer(256, layers[2], stride=1, dilation=2, norm_layer=norm_layer, bn_mom=bn_mom)
            self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=4, norm_layer=norm_layer, bn_mom=bn_mom,
                                           multi_grid=True)
        else:
            self.layer3 = self._make_layer(256, layers[2], stride=2, dilation=1, norm_layer=norm_layer, bn_mom=bn_mom)
            self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=2, norm_layer=norm_layer, bn_mom=bn_mom,
                                           multi_grid=True)

        self._out_feature_strides = {"res5": stride}
        self._out_feature_channels = {"res5": 512 * Bottleneck.expansion}
        self._out_features = ["res5"]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_msra_fill(m)

    def _make_layer(self, planes, blocks, stride=1, dilation=1, norm_layer='BN', bn_mom=0.1, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * Bottleneck.expansion,
                       kernel_size=1, stride=stride, bias=False),
                get_norm(norm_layer, planes * Bottleneck.expansion, bn_mom))

        layers = []
        multi_dilations = [1, 2, 4]
        if multi_grid:
            layers.append(
                Bottleneck(self.inplanes, planes, stride, dilation=multi_dilations[0] * dilation, downsample=downsample,
                           norm_layer=norm_layer, bn_mom=bn_mom))
        elif dilation == 1 or dilation == 2:
            layers.append(
                Bottleneck(self.inplanes, planes, stride, dilation=1, downsample=downsample, norm_layer=norm_layer,
                           bn_mom=bn_mom))
        elif dilation == 4:
            layers.append(
                Bottleneck(self.inplanes, planes, stride, dilation=2, downsample=downsample, norm_layer=norm_layer,
                           bn_mom=bn_mom))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(
                    Bottleneck(self.inplanes, planes, dilation=multi_dilations[i] * dilation, norm_layer=norm_layer,
                               bn_mom=bn_mom))
            else:
                layers.append(
                    Bottleneck(self.inplanes, planes, dilation=dilation, norm_layer=norm_layer, bn_mom=bn_mom))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return {'res5': x}


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
    bn_mom = cfg.MODEL.EMA.BN_MOM
    # fmt: on

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]
    return DilatedResNet(num_blocks_per_stage, stride, norm, bn_mom)
