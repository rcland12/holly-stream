"""
Export an Ultralytics YOLO detection model to an ONNX file that DeepStream's
nvinfer (TensorRT 8.2 on JetPack 4.6) can consume directly.

Run this on any x86 machine with `pip install ultralytics onnx onnxslim`; the
Jetson itself cannot run modern Ultralytics. models/train.py calls it
automatically after training. Copy the resulting .onnx to
models/ on the Jetson and the container builds the TensorRT engine on first
start.

The exported graph has a single output named "output" with shape
[1, N, 6] where each row is [x1, y1, x2, y2, score, class_id] in network-input
pixel coordinates. That is exactly what app/src/nvdsparsebbox_yolo.cpp reads.

N is the number of anchors (e.g. 5040 for 640x384) and rows are unfiltered;
nvinfer applies the confidence threshold.

  * Anchor-based heads (YOLOv8 / YOLO11): duplicates remain, keep NMS on
    (NMS_IOU=0.45).
  * End-to-end heads (YOLO26): the one-to-one head gives one box per object,
    so NMS can be turned off (NMS_IOU=0).

Non-square inputs are supported and recommended: a 16:9 camera letterboxed into
640x384 keeps the same horizontal resolution as 640x640 for ~60% of the compute.

Alongside the model it writes <model>_labels.txt with the class names from the
weights, which the app picks up automatically (custom models need no extra
setup).

Usage:
  python export.py --weights yolo26n.pt --width 640 --height 384
  python export.py --weights runs/detect/train/weights/best.pt --width 640 --height 384 --output holly_640x384.onnx
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class DeepStreamOutput(nn.Module):
    """Convert raw anchor predictions [B, 4 + nc, N] (cx, cy, w, h, scores...) into [B, N, 6]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        cxcy, wh, scores = x[..., 0:2], x[..., 2:4], x[..., 4:]
        score, label = scores.max(dim=-1, keepdim=True)
        return torch.cat([cxcy - wh / 2, cxcy + wh / 2, score, label.to(x.dtype)], dim=-1)


class EndToEndOutput(nn.Module):
    """Convert one-to-one head predictions [B, N, 4 + nc] (x1, y1, x2, y2, scores...) into [B, N, 6].

    Ultralytics' own end-to-end post-processing uses TopK + Mod, and TensorRT 8.2
    cannot import Mod. The one-to-one head already yields a single box per object,
    so emitting every anchor and letting nvinfer apply the confidence threshold
    gives the same detections without those ops.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score, label = x[..., 4:].max(dim=-1, keepdim=True)
        return torch.cat([x[..., :4], score, label.to(x.dtype)], dim=-1)


def load_model(weights: str, end2end: bool) -> tuple[nn.Module, list[str]]:
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    model = YOLO(weights).model.float().eval()
    names = model.names if isinstance(model.names, list) else [model.names[i] for i in sorted(model.names)]
    for m in model.modules():
        if isinstance(m, Detect):
            has_one2one = getattr(m, "one2one_cv2", None) is not None
            m.end2end = end2end and has_one2one
            m.export = True
            m.format = "onnx"
            m.dynamic = False
    model = model.fuse()

    head = next(m for m in model.modules() if isinstance(m, Detect))
    if head.end2end:
        print(f"{weights}: end-to-end head (NMS-free), run with NMS_IOU=0")
        head.postprocess = lambda preds: preds  # keep all anchors, see EndToEndOutput
        return nn.Sequential(model, EndToEndOutput()), names
    print(f"{weights}: anchor head, DeepStream will run NMS")
    return nn.Sequential(model, DeepStreamOutput()), names


def export_onnx(weights: str, width: int = 640, height: int = 384, output: str | None = None,
                opset: int = 12, end2end: bool = True) -> Path:
    """Export `weights` to a DeepStream-ready ONNX file plus <model>_labels.txt. Returns the ONNX path."""
    import onnx
    import onnxslim

    for size in (width, height):
        if size % 32:
            raise SystemExit(f"Input size {size} is not a multiple of 32")

    model, names = load_model(weights, end2end=end2end)
    dummy = torch.zeros(1, 3, height, width)
    with torch.no_grad():
        out = model(dummy)
    print(f"output shape: {list(out.shape)}")

    output = Path(output or f"{Path(weights).stem}_{width}x{height}.onnx")
    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        do_constant_folding=True,
        dynamo=False,
    )

    slim = onnxslim.slim(onnx.load(str(output)))
    onnx.checker.check_model(slim)
    onnx.save(slim, str(output))
    print(f"saved {output}")

    # The app reads <model>_labels.txt next to the ONNX (falls back to models/labels.txt),
    # so a custom model's class names travel with it.
    labels = output.with_name(f"{output.stem}_labels.txt")
    labels.write_text("\n".join(names) + "\n")
    print(f"saved {labels} ({len(names)} classes)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Ultralytics .pt file (downloaded automatically if a known name)")
    parser.add_argument("--width", type=int, default=640, help="network input width, multiple of 32")
    parser.add_argument("--height", type=int, default=384, help="network input height, multiple of 32")
    parser.add_argument("--opset", type=int, default=12, help="TensorRT 8.2 supports opset <= 13")
    parser.add_argument("--no-end2end", action="store_true", help="use the one-to-many head even if the model has an NMS-free head")
    parser.add_argument("--output", help="output path (default: <weights>_<W>x<H>.onnx)")
    args = parser.parse_args()

    export_onnx(args.weights, args.width, args.height, args.output, args.opset, end2end=not args.no_end2end)


if __name__ == "__main__":
    main()
