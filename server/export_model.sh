#!/bin/bash
#
# Exports a YOLO detection model to ONNX for the Holly detector and puts it in server/models/.
#
# Usage: ./export_model.sh [weights] [height] [width]
#   weights   a stock name (yolo26m.pt, yolo26s.pt, yolo11s.pt, ...) which is downloaded, or your own best.pt (default yolo26m.pt)
#   height    model input height (default 480)
#   width     model input width (default 640)
#
# Requires: pip install ultralytics onnx onnxslim   (CPU is fine; exporting does not need a GPU)
#
# The detector expects one [1, 3, H, W] input and one [1, 300, 6] output of [x1, y1, x2, y2, score, class].
# YOLO26 produces that natively (it is NMS-free); older models such as YOLO11 get NMS built into the graph.
# The detector builds a TensorRT engine from the ONNX file on its first start, which takes a few minutes.
#
# The detector feeds the model half-resolution frames, letterboxed. 480x640 takes a 1280x960 camera with no resize and a
# 1280x720 webcam with no resize and 60-pixel bars; 384x640 suits 16:9 cameras only (both sides multiples of 32).

set -euo pipefail

WEIGHTS="${1:-yolo26m.pt}"
HEIGHT="${2:-480}"
WIDTH="${3:-640}"

if (( HEIGHT % 32 || WIDTH % 32 )); then
    echo "Height and width must be multiples of 32 (got ${HEIGHT}x${WIDTH})." >&2
    exit 1
fi
if [[ -f "${WEIGHTS}" ]]; then
    WEIGHTS="$(realpath "${WEIGHTS}")"
elif [[ "${WEIGHTS}" == */* ]]; then
    echo "Weights file not found: ${WEIGHTS}" >&2
    exit 1
fi

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"
mkdir -p "${MODELS_DIR}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
cd "${WORK_DIR}"

python3 - "${WEIGHTS}" "${HEIGHT}" "${WIDTH}" "${MODELS_DIR}" <<'EOF'
import shutil
import sys
from pathlib import Path

import onnx
from ultralytics import YOLO

weights, height, width, models_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])


def output_shape(path):
    output = onnx.load(path, load_external_data=False).graph.output[0]
    return [d.dim_value for d in output.type.tensor_type.shape.dim]


def export(nms):
    model = YOLO(weights)
    if model.task != "detect":
        sys.exit(f"{weights} is a '{model.task}' model; only detection models are supported.")
    return model.export(format="onnx", imgsz=(height, width), batch=1, dynamic=False, simplify=True,
                        opset=17, nms=nms, conf=0.25, iou=0.45)


# End-to-end models (YOLO26) already output final detections; everything else needs NMS in the graph
path = export(nms=False)
if output_shape(path) != [1, 300, 6]:
    path = export(nms=True)
if output_shape(path) != [1, 300, 6]:
    sys.exit(f"Unexpected output shape {output_shape(path)}; expected [1, 300, 6].")

destination = models_dir / f"{Path(weights).stem}_{height}x{width}.onnx"
shutil.move(path, destination)
names = YOLO(weights).names
print(f"\nSaved {destination}")
print("Classes: " + ", ".join(f"{i}={names[i]}" for i in range(len(names))))
print(f"Use it with MODEL=/models/{destination.name} in server/.env, and set CLASSES to the indexes to draw.")
EOF
