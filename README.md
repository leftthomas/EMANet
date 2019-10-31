# EMANet
A PyTorch implementation of EMANet based on ICCV 2019 paper [Expectation-Maximization Attention Networks for Semantic Segmentation](https://arxiv.org/abs/1907.13426). 

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
- panopticapi
```
pip install git+https://github.com/cocodataset/panopticapi.git
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

### Expected dataset structure for COCO:
```
coco/
  annotations/
    panoptic_{train,val}2017.json
  panoptic_{train,val}2017/
  # png annotations
  panoptic_stuff_{train,val}2017/  # generated by the script mentioned below
```
run `./datasets/prepare_coco.py` to extract semantic annotations from panoptic annotations.

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

For example, to launch end-to-end EMANet training with `ResNet-50` backbone for `coco` dataset on 8 GPUs, one should execute:
```bash
python train_net.py --config-file configs/r50_coco.yaml --num-gpus 8
```

## Evaluation
Model evaluation can be done similarly:
```bash
python train_net.py --config-file configs/r50_coco.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS epochs/model.pth
```

## Results
There are some difference between this implementation and official implementation:
1. The image sizes of `Multi-Scale Training` are (640, 672, 704, 736, 768, 800) for `coco` dataset;
2. The image sizes of `Multi-Scale Training` are (800, 832, 864, 896, 928, 960, 992, 1024) for `cityscapes` dataset;
3. Learning rate policy is `WarmupCosineLR`;

### COCO
<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Name</th>
		<th>train time (s/iter)</th>
		<th>inference time (s/im)</th>
		<th>train mem (GB)</th>
		<th>PA</br>%</th>
		<th>mean PA %</th>
		<th>mean IoU %</th>
		<th>FW IoU %</th>
		<th>download link</th>
		<!-- TABLE BODY -->
		<!-- ROW: r50 -->
		<tr>
			<td align="center"><a href="configs/r50_coco.yaml">R50</a></td>
			<td align="center">0.49</td>
			<td align="center">0.12</td>
			<td align="center">27.12</td>
			<td align="center">94.19</td>
			<td align="center">75.31</td>
			<td align="center">66.64</td>
			<td align="center">89.54</td>
			<td align="center"><a href="https://pan.baidu.com/s/18wRQbLQyqXA4ISloUGWTSA">model</a>&nbsp;|&nbsp;ga7k</td>
		</tr>
		<!-- ROW: r101 -->
		<tr>
			<td align="center"><a href="configs/r101_coco.yaml">R101</a></td>
			<td align="center">0.65</td>
			<td align="center">0.16</td>
			<td align="center">28.81</td>
			<td align="center">94.29</td>
			<td align="center">76.08</td>
			<td align="center">67.57</td>
			<td align="center">89.69</td>
			<td align="center"><a href="https://pan.baidu.com/s/1eqt2U2gIBeE_UMtluCKIcQ">model</a>&nbsp;|&nbsp;xnvs</td>
		</tr>
		<!-- ROW: r152 -->
		<tr>
			<td align="center"><a href="configs/r152_coco.yaml">R152</a></td>
			<td align="center">0.65</td>
			<td align="center">0.16</td>
			<td align="center">28.81</td>
			<td align="center">94.29</td>
			<td align="center">76.08</td>
			<td align="center">67.57</td>
			<td align="center">89.69</td>
			<td align="center"><a href="https://pan.baidu.com/s/1eqt2U2gIBeE_UMtluCKIcQ">model</a>&nbsp;|&nbsp;xnvs</td>
		</tr>
	</tbody>
</table>

### Cityscapes
<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Name</th>
		<th>train time (s/iter)</th>
		<th>inference time (s/im)</th>
		<th>train mem (GB)</th>
		<th>PA</br>%</th>
		<th>mean PA %</th>
		<th>mean IoU %</th>
		<th>FW IoU %</th>
		<th>download link</th>
		<!-- TABLE BODY -->
		<!-- ROW: r50 -->
		<tr>
			<td align="center"><a href="configs/r50_cityscapes.yaml">R50</a></td>
			<td align="center">0.49</td>
			<td align="center">0.12</td>
			<td align="center">27.12</td>
			<td align="center">94.19</td>
			<td align="center">75.31</td>
			<td align="center">66.64</td>
			<td align="center">89.54</td>
			<td align="center"><a href="https://pan.baidu.com/s/18wRQbLQyqXA4ISloUGWTSA">model</a>&nbsp;|&nbsp;ga7k</td>
		</tr>
		<!-- ROW: r101 -->
		<tr>
			<td align="center"><a href="configs/r101_cityscapes.yaml">R101</a></td>
			<td align="center">0.65</td>
			<td align="center">0.16</td>
			<td align="center">28.81</td>
			<td align="center">94.29</td>
			<td align="center">76.08</td>
			<td align="center">67.57</td>
			<td align="center">89.69</td>
			<td align="center"><a href="https://pan.baidu.com/s/1eqt2U2gIBeE_UMtluCKIcQ">model</a>&nbsp;|&nbsp;xnvs</td>
		</tr>
		<!-- ROW: r152 -->
		<tr>
			<td align="center"><a href="configs/r152_cityscapes.yaml">R152</a></td>
			<td align="center">0.65</td>
			<td align="center">0.16</td>
			<td align="center">28.81</td>
			<td align="center">94.29</td>
			<td align="center">76.08</td>
			<td align="center">67.57</td>
			<td align="center">89.69</td>
			<td align="center"><a href="https://pan.baidu.com/s/1eqt2U2gIBeE_UMtluCKIcQ">model</a>&nbsp;|&nbsp;xnvs</td>
		</tr>
	</tbody>
</table>