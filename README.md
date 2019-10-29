# DANet
A PyTorch implementation of DANet based on CVPR 2019 paper [Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983). 

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
- opencv
```
pip install opencv-python
```
- tensorboard
```
pip install tensorboard
```
- pycocotools
```
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```
- fvcore
```
pip install git+https://github.com/facebookresearch/fvcore
```
- cityscapesScripts
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```
- detectron2
```
pip install git+https://github.com/facebookresearch/detectron2.git@master
```

## Datasets
For a few datasets that detectron2 natively supports, the datasets are assumed to exist in a directory called
`datasets/`, under the directory where you launch the program. They need to have the following directory structure:

### Expected dataset structure for Cityscapes:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```
run `./datasets/prepare_cityscapes.py` to creat `labelTrainIds.png`.

## Training
To train a model, run
```bash
python train_net.py --config-file <config.yaml>
```

For example, to launch end-to-end DANet training with ResNet-50 backbone on 8 GPUs, one should execute:
```bash
python train_net.py --config-file configs/r50.yaml --num-gpus 8
```

## Evaluation
Model evaluation can be done similarly:
```bash
python train_net.py --config-file configs/r50.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS epochs/model.pth
```

## Results
There are some difference between this implementation and official implementation:
1. No `Multi-Grid` and `Multi-Scale Testing`;
2. The image sizes of `Multi-Scale Training` are (800, 832, 864, 896, 928, 960);
3. Training step is set to `24000`;
4. Learning rate policy is `WarmupMultiStepLR`;
5. `Position Attention Module (PAM)` use the similar mechanism as `Channel Attention Module (CAM)`, just use the tensor
and its transpose to compute the attention. 


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">kp.<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: keypoint_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/r50_fpn.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.315</td>
<td align="center">0.102</td>
<td align="center">5.0</td>
<td align="center">53.6</td>
<td align="center">64.0</td>
<td align="center">137261548</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x/137261548/model_final_04e291.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x/137261548/metrics.json">metrics</a></td>
</tr>
</tbody></table>