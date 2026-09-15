#!/bin/bash

# Instructions:
#
# Converts a YOLO11 detection model (the stock yolo11n.pt or your own trained best.pt) to ONNX with NMS
# built into the graph, and installs it into the Triton model repository for Triton's onnxruntime backend (CPU).
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
# Usage: ./export.sh [weights] [height] [width]
#   weights   .pt file to convert (default yolo11n.pt, downloaded if missing)
#   height    model input height (default 256)
#   width     model input width (default 320)
#   CONF=0.25 IOU=0.45 ./export.sh ...   override the thresholds baked into the graph
#
# Examples:
#   ./export.sh                                               # stock COCO model, 256x320
#   ./export.sh ~/runs/detect/holly/weights/best.pt           # your own model, 256x320
#   ./export.sh ~/runs/detect/holly/weights/best.pt 384 512   # your own model, larger input
#
# The Pi camera modules are 4:3, so a square input wastes a quarter of the compute on letterbox
# padding. Measured on a Pi 4 while streaming 1280x960@30, with YOLO11n COCO val2017 accuracy:
#   height width   inference   person AP50-95 (small / medium / large objects)
#   640    640     ~1000 ms    51.8 (28.9 / 62.2 / 78.1)
#   384    512     ~500 ms     48.5 (23.1 / 58.4 / 77.9)
#   256    320     ~210 ms     39.2 (12.3 / 46.7 / 74.1)   default, fine for subjects that fill part of the frame

set -euo pipefail

WEIGHTS="${1:-yolo11n.pt}"
HEIGHT="${2:-256}"
WIDTH="${3:-320}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"

if (( HEIGHT % 32 || WIDTH % 32 )); then
    echo "Height and width must be multiples of 32 (got ${HEIGHT}x${WIDTH})." >&2
    exit 1
fi

# Resolve the weights against the caller's directory before changing into triton/.
# A bare name that doesn't exist (e.g. yolo11n.pt) is left as is so ultralytics downloads it.
if [[ -f "${WEIGHTS}" ]]; then
    WEIGHTS="$(realpath "${WEIGHTS}")"
elif [[ "${WEIGHTS}" == */* ]]; then
    echo "Weights file not found: ${WEIGHTS}" >&2
    exit 1
fi

cd "$(dirname "$0")"

python3 - "${WEIGHTS}" "${HEIGHT}" "${WIDTH}" "${CONF}" "${IOU}" <<'EOF'
import shutil
import sys

import onnx
from ultralytics import YOLO

weights, height, width, conf, iou = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

model = YOLO(weights)
if model.task != "detect":
    sys.exit(f"{weights} is a '{model.task}' model; only detection models are supported.")

onnx_path = model.export(
    format="onnx",
    imgsz=(height, width),
    batch=1,
    dynamic=False,
    simplify=True,
    opset=17,
    nms=True,
    conf=conf,
    iou=iou,
)
shutil.move(onnx_path, "repository/object_detection/1/model.onnx")

# main.py and test.py read class names from here, so they must follow the model's class indexes
names = model.names
with open("repository/yolo11/labels.txt", "w") as file:
    file.writelines(f"{names[i]}\n" for i in range(len(names)))

# Catch a mismatch here rather than as a Triton load error on the Pi
graph = onnx.load("repository/object_detection/1/model.onnx").graph
shapes = {
    tensor.name: [d.dim_value for d in tensor.type.tensor_type.shape.dim]
    for tensor in (*graph.input, *graph.output)
}
expected = {"images": [1, 3, height, width], "output0": [1, 300, 6]}
if shapes != expected:
    sys.exit(f"Exported model has {shapes}, expected {expected}.")

print(f"\nClasses ({len(names)}): " + ", ".join(f"{i}={names[i]}" for i in range(len(names))))
EOF

# The preprocess letterbox size and the ONNX input size must match the export size
sed -i -E "s/dims: \[ 1, 3, [0-9]+, [0-9]+ \]/dims: [ 1, 3, ${HEIGHT}, ${WIDTH} ]/" \
    repository/object_detection/config.pbtxt \
    repository/preprocess/config.pbtxt

echo "Installed repository/object_detection/1/model.onnx at ${HEIGHT}x${WIDTH} (HxW) and repository/yolo11/labels.txt."
echo "If the classes changed, update CLASSES in .env to the new indexes above (or CLASSES=\"[]\" for all)."
echo "Then copy triton/repository to the Pi and restart: docker compose restart triton app"
