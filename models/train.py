"""
Train a custom YOLO model on your own labeled images and export it for the Jetson.

Run on a machine with an NVIDIA GPU (training on CPU works but is very slow):

  pip install ultralytics onnx onnxslim

  python models/train.py --dataset data/labeled --name holly

--dataset is a YOLO-format export from a labeling tool. These layouts all work:

  Label Studio "YOLO" export      images/*.jpg  labels/*.txt  classes.txt
  CVAT "YOLO 1.1" (with images)   obj_train_data/*.jpg + *.txt  obj.names
  Plain folder                    *.jpg next to *.txt, class names via --names
  Roboflow / Ultralytics export   data.yaml with its own train/valid split

Each label file has one line per box, "class x_center y_center width height",
normalized to 0-1. Images without a label file are kept as background images
(frames where nothing should be detected), which reduces false positives.

The script splits the images into train/val (unless data.yaml already does),
trains from a pretrained checkpoint (results, plots and weights go to
runs/<name>/; re-running with the same name overwrites it), then exports the
best weights with models/export.py to:

  models/<name>_<W>x<H>.onnx
  models/<name>_<W>x<H>_labels.txt

Copy both files to models/ on the Jetson and set MODEL=<name>_<W>x<H>.onnx.
"""

import argparse
import random
import sys
from pathlib import Path

import yaml

MODELS_DIR = Path(__file__).resolve().parent
REPO_DIR = MODELS_DIR.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
NAME_FILES = ("classes.txt", "obj.names", "names.txt")


def label_path(image: Path) -> Path:
    """Where Ultralytics looks for an image's labels: .../images/x.jpg -> .../labels/x.txt, else x.txt beside it."""
    parts = list(image.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def find_names(dataset: Path, names_arg: str | None) -> list[str]:
    if names_arg:
        return [name.strip() for name in names_arg.split(",") if name.strip()]
    for filename in NAME_FILES:
        for path in sorted(dataset.rglob(filename)):
            names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            if names:
                print(f"class names from {path}: {names}")
                return names
    sys.exit(f"No class names found in {dataset} ({', '.join(NAME_FILES)}); pass --names person,dog")


def prepare_split(dataset: Path, names: list[str], val_fraction: float, seed: int, run_dir: Path) -> Path:
    """Write train.txt / val.txt image lists and a data.yaml for a dataset without its own split."""
    images = sorted(p.resolve() for p in dataset.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        sys.exit(f"No images found under {dataset}")

    labeled, background, boxes, bad = [], [], 0, 0
    for image in images:
        labels = label_path(image)
        if not labels.exists():
            background.append(image)
            continue
        labeled.append(image)
        for line in labels.read_text().splitlines():
            if not line.strip():
                continue
            boxes += 1
            if not 0 <= int(float(line.split()[0])) < len(names):
                bad += 1
    print(f"{len(images)} images: {len(labeled)} labeled ({boxes} boxes), {len(background)} background")
    if bad:
        sys.exit(f"{bad} boxes use a class id outside 0-{len(names) - 1}; check the class names file")
    if not labeled:
        sys.exit("No label files found next to the images (expected labels/<image>.txt or <image>.txt)")
    if len(background) > len(labeled):
        print("warning: more background images than labeled ones; is the labels folder in the right place?")

    rng = random.Random(seed)
    val, train = [], []
    # Split labeled and background images separately so both sets get some of each.
    for group in (labeled, background):
        group = group[:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        val += group[:n_val]
        train += group[n_val:]

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train.txt").write_text("\n".join(map(str, train)) + "\n")
    (run_dir / "val.txt").write_text("\n".join(map(str, val)) + "\n")
    data_yaml = run_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "train": str(run_dir / "train.txt"),
        "val": str(run_dir / "val.txt"),
        "names": dict(enumerate(names)),
    }, sort_keys=False))
    print(f"split: {len(train)} train / {len(val)} val -> {data_yaml}")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path, help="YOLO-format dataset folder (see above)")
    parser.add_argument("--name", default="custom", help="model name, used for the run folder and ONNX file")
    parser.add_argument("--names", help="comma-separated class names, if the dataset has no classes file")
    parser.add_argument("--model", default="yolo26n.pt", help="pretrained checkpoint to start from (yolo26n.pt, yolo26s.pt, yolo11n.pt, ...)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="training image size")
    parser.add_argument("--batch", type=int, default=16, help="batch size (-1 = auto-fit GPU memory)")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu (default: GPU if available)")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="share of images held out for validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640, help="exported network width (match the camera aspect ratio)")
    parser.add_argument("--height", type=int, default=384, help="exported network height")
    parser.add_argument("--no-export", action="store_true", help="only train")
    args = parser.parse_args()

    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    sys.path.insert(0, str(MODELS_DIR))
    from export import export_onnx

    dataset = args.dataset.resolve()
    run_dir = REPO_DIR / "runs" / args.name
    if (dataset / "data.yaml").exists():
        data_yaml = dataset / "data.yaml"
        print(f"using the dataset's own split: {data_yaml}")
    else:
        data_yaml = prepare_split(dataset, find_names(dataset, args.names), args.val_fraction, args.seed, run_dir)

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(run_dir.parent),
        name=args.name,
        exist_ok=True,
        seed=args.seed,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")
    if args.no_export:
        return

    onnx_path = export_onnx(str(best), args.width, args.height, str(MODELS_DIR / f"{args.name}_{args.width}x{args.height}.onnx"))
    head = next(m for m in YOLO(str(best)).model.modules() if isinstance(m, Detect))
    nms_iou = 0 if getattr(head, "one2one_cv2", None) is not None else 0.45

    print(f"""
Done. Deploy on the Jetson:

  scp {onnx_path} {onnx_path.with_name(onnx_path.stem + '_labels.txt')} <jetson>:~/dev/holly-stream/models/

  # .env
  MODEL={onnx_path.name}
  NMS_IOU={nms_iou}
  CLASSES=

  ./run.sh                   # recreates the app; first start builds the TensorRT engine (10-15 min)
""")


if __name__ == "__main__":
    main()
