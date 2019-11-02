from detectron2.config import CfgNode as CN


def add_ema_config(cfg):
    """
    Add config for DANet.
    """
    _C = cfg

    _C.MODEL.DILATED_RESNET = CN()

    _C.MODEL.DILATED_RESNET.DEPTH = 50
    _C.MODEL.DILATED_RESNET.NORM = "FrozenBN"
    _C.MODEL.DILATED_RESNET.STRIDE = 8
