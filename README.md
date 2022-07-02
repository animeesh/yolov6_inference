# YOLOv6 

## Introduction

YOLOv6 is a single-stage object detection framework dedicated to industrial applications, with hardware-friendly efficient design and high performance.

<img src="assets/picture.png" width="800">

YOLOv6-nano achieves 35.0 mAP on COCO val2017 dataset with 1242 FPS on T4 using TensorRT FP16 for bs32 inference, and YOLOv6-s achieves 43.1 mAP on COCO val2017 dataset with 520 FPS on T4 using TensorRT FP16 for bs32 inference.

YOLOv6 is composed of the following methods:

- Hardware-friendly Design for Backbone and Neck
- Efficient Decoupled Head with SIoU Loss

## Quick Start

### Install

```shell
git clone https://github.com/animeesh/yolov6_inferencing
cd yolov6_inferencing
pip install -r requirements.txt
```

### Inference



```shell
python infer.py 
```


