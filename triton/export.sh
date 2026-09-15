#!/bin/bash

# Instructions:
#
# Exports YOLO11n to ONNX with NMS built into the graph for Triton's onnxruntime backend (CPU).
# Run this on a separate machine (not the Pi) with: pip install ultralytics onnx onnxslim
# Tested with ultralytics 8.4.150.
#
# Unlike the jetson branch there is no TensorRT step. The Pi has no NVIDIA GPU, and nms=True exports
# the standard ONNX NonMaxSuppression op, which ONNX Runtime runs on CPU. opset 17 keeps it loadable
# by the onnxruntime version shipped in tritonserver:24.01.
#
# The model outputs output0 [1, 300, 6] as [x1, y1, x2, y2, score, class] in model input pixels,
# zero-padded to 300 rows. conf/iou are baked into the graph.
#
# Usage: ./export.sh [imgsz]   (default 640; 320 is ~4x faster on a Pi 4 but misses more small objects)

set -euo pipefail

IMGSZ="${1:-640}"
cd "$(dirname "$0")"

yolo export \
    model=yolo11n.pt \
    format=onnx \
    imgsz="${IMGSZ}" \
    batch=1 \
    dynamic=False \
    simplify=True \
    opset=17 \
    nms=True \
    conf=0.25 \
    iou=0.45

mv yolo11n.onnx repository/object_detection/1/model.onnx

# The preprocess letterbox size and the ONNX input size must match the export size
sed -i -E "s/dims: \[ 1, 3, [0-9]+, [0-9]+ \]/dims: [ 1, 3, ${IMGSZ}, ${IMGSZ} ]/" \
    repository/object_detection/config.pbtxt \
    repository/preprocess/config.pbtxt

echo "Exported repository/object_detection/1/model.onnx at ${IMGSZ}x${IMGSZ}."
