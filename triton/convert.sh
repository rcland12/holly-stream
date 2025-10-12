#!/bin/bash

# Instructions:
#
# You will most likely have to run the first two steps on a seperate server that can run a newer version of python and can actually install all the dependencies.
# The python script requires the onnx package the onnx_graphsurgeon.
#
# 1) This first command will convert your PyTorch model to ONNX. This model will have nodes that are incompatible with tensorrt.
# 2) The convert.py script will take the incompatible nodes and remove them or convert them as necessary. It will outputs another ONNX model.
# 3) The last command will convert your new, second ONNX model to the final model in tensorrt format (.plan).

yolo export \
    model=yolo11n.pt \
    format=onnx \
    imgsz=640 \
    batch=1 \
    dynamic=False \
    simplify=True \
    opset=12 \
    half=True \
    conf=0.5 \
    iou=0.45 \
    nms=True

python convert.py --in yolo11n.onnx --out model.onnx --scores_are_logits

/usr/src/tensorrt/bin/trtexec \
    --onnx=./model.onnx \
    --saveEngine=./model.plan \
    --workspace=2048 \
    --fp16 \
    --verbose
