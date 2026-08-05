#!/bin/bash

# yolo export \
#     model=model.pt \
#     format=onnx \
#     imgsz=640 \
#     batch=1 \
#     dynamic=False \
#     opset=12 \
#     half=True \
#     nms=False

# yolo export \
#     model=yolo26n.pt \
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

# docker run --rm -it --gpus '"device=0"' -v ./:/models nvcr.io/nvidia/tensorrt:24.04-py3 \
#     trtexec \
#         --onnx=/models/model_nms.onnx \
#         --saveEngine=/models/model.plan \
#         --workspace=4096 \
#         --fp16 \
#         --device=0 \
#         --useCudaGraph \
#         --timingCacheFile=/tmp/timing.cache
