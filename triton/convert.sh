#!/bin/bash

# yolo export \
#     model=yolo11n.pt \
#     format=onnx \
#     imgsz=640 \
#     batch=1 \
#     dynamic=False \
#     simplify=True \
#     opset=12 \
#     half=True \
#     conf=0.5 \
#     iou=0.45 \
#     nms=True

yolo export \
	model=yolo11n.pt \
	format=onnx \
    imgsz=640 \
	simplify=True \
    dynamic=True \
    half=True \
	opset=20 \
    conf=0.5 \
    iou=0.45 \
    nms=True \
	device=0

# docker run --rm -it --gpus '"device=0"' -v ./:/models nvcr.io/nvidia/tensorrt:24.04-py3 \
#     trtexec \
#         --onnx=/models/new_yolo.onnx \
#         --saveEngine=/models/model.plan \
#         --workspace=4096 \
#         --fp16 \
#         --device=0 \
#         --useCudaGraph \
#         --timingCacheFile=/tmp/timing.cache


# /usr/src/tensorrt/bin/trtexec \
#     --onnx=./model.onnx \
#     --saveEngine=./model.plan \
#     --workspace=2048 \
#     --fp16 \
#     --verbose
