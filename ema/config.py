from detectron2.config import CfgNode as CN


def add_ema_config(cfg):
    """
    Add config for EMANet.
    """
    _C = cfg

    _C.MODEL.DILATED_RESNET = CN()

    _C.MODEL.DILATED_RESNET.DEPTH = 50
    _C.MODEL.DILATED_RESNET.NORM = "BN"
    _C.MODEL.DILATED_RESNET.STRIDE = 8
    _C.MODEL.SEM_SEG_HEAD.ITERATION_NUM = 3

    _C.MODEL.EMA = CN()
    _C.MODEL.EMA.EM_MOM = 0.9
    _C.MODEL.EMA.BN_MOM = 3e-4
