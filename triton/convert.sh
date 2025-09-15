#!/bin/bash

yolo export \
	model=yolo11n.pt \
	format=onnx \
	simplify=True \
    dynamic=True \
	opset=20 \
	device=0

docker run --rm -it --gpus '"device=0"' -v ./:/models nvcr.io/nvidia/tensorrt:24.04-py3 \
    trtexec \
        --onnx=/models/yolo11n.onnx \
        --saveEngine=/models/model.plan \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:8x3x640x640 \
        --workspace=4096 \
        --useCudaGraph \
        --timingCacheFile=/tmp/timing.cache \
        --calib=/models/yolo11n-int8.cache
