import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def tmap_all(g: gs.Graph) -> Dict[str, gs.Variable]:
    """Map every tensor name (inputs, intermediate, outputs) to a gs.Variable."""
    m = {t.name: t for t in g.tensors().values()}
    for n in g.nodes:
        for o in n.outputs:
            m.setdefault(o.name, o)
    for o in g.outputs:
        m.setdefault(o.name, o)
    for i in g.inputs:
        m.setdefault(i.name, i)
    return m


def try_find_boxes_scores_from_slices(
    g: gs.Graph,
) -> Tuple[Optional[gs.Variable], Optional[gs.Variable], str]:
    """
    Look for typical YOLO decoded outputs:
      boxes:  Slice -> (1, N, 4)
      scores: Slice -> (1, N, 80)
    Returns (boxes, scores, dbg). If not found, returns (None, None, reason).
    """
    m = tmap_all(g)
    boxes, scores = None, None
    for n in g.nodes:
        if n.op != "Slice":
            continue
        if not n.outputs or len(n.outputs) != 1:
            continue
        out = n.outputs[0]
        shp = getattr(out, "shape", None)
        if not (isinstance(shp, (list, tuple)) and len(shp) == 3):
            continue
        if shp[2] == 4 and boxes is None:
            boxes = out
        elif shp[2] == 80 and scores is None:
            scores = out
    if boxes is not None and scores is not None:
        dbg = f"found Slice outputs: boxes={boxes.name} {boxes.shape}, scores={scores.name} {scores.shape}"
        return boxes, scores, dbg

    cand_boxes = [
        m[k]
        for k in m
        if k.endswith("/Slice_output_0") or k.endswith("Slice_output_0")
    ]
    cand_scores = [
        m[k]
        for k in m
        if k.endswith("/Slice_1_output_0") or k.endswith("Slice_1_output_0")
    ]
    if cand_boxes and cand_scores:
        return cand_boxes[0], cand_scores[0], "fallback by common YOLO names"

    return None, None, "no (1,N,4) and (1,N,80) Slice outputs found"


def find_head_84_and_slice(
    g: gs.Graph,
) -> Tuple[gs.Variable, gs.Variable, str]:
    """
    Fallback: find decoded head with 84 channels in BCN [1,84,N] or BNC [1,N,84],
    convert to BNC, then slice out boxes [1,N,4] and scores [1,N,80].
    """

    def is3(s):
        return isinstance(s, (list, tuple)) and len(s) == 3

    m = tmap_all(g)
    bcn, bnc = [], []
    for name, t in m.items():
        s = getattr(t, "shape", None)
        if not is3(s):
            continue
        if s[1] == 84:
            bcn.append((name, t, s))
        if s[2] == 84:
            bnc.append((name, t, s))

    def pref_key(item):
        name, _, _ = item
        score = 0
        if "Concat" in name:
            score -= 2
        if "/model/" in name or "model." in name:
            score -= 1
        return (score, name)

    bcn.sort(key=pref_key)
    bnc.sort(key=pref_key)

    if bcn:
        name, head_bcn, s = bcn[0]
        head_bnc = gs.Variable("head_bnc", dtype=np.float32, shape=[1, -1, 84])
        g.nodes.append(
            gs.Node(
                op="Transpose",
                name="node_head_transpose",
                inputs=[head_bcn],
                outputs=[head_bnc],
                attrs={"perm": [0, 2, 1]},
            )
        )
        src = f"head BCN {name} {s} -> transpose -> [1,N,84]"
    elif bnc:
        name, head_bnc, s = bnc[0]
        src = f"head BNC {name} {s}"
    else:
        for n in g.nodes:
            if n.op == "Concat" and n.outputs:
                name = n.outputs[0].name
                head_bcn = m[name]
                head_bnc = gs.Variable(
                    "head_bnc", dtype=np.float32, shape=[1, -1, 84]
                )
                g.nodes.append(
                    gs.Node(
                        op="Transpose",
                        name="node_head_transpose_fb",
                        inputs=[head_bcn],
                        outputs=[head_bnc],
                        attrs={"perm": [0, 2, 1]},
                    )
                )
                src = f"fallback Concat {name} -> transpose -> [1,N,84]"
                break
        else:
            raise RuntimeError(
                "Could not locate a decoded head with 84 channels."
            )

    def const_i32(name, val):
        return gs.Constant(name=name, values=np.array(val, dtype=np.int32))

    boxes = gs.Variable("boxes_bnc", dtype=np.float32, shape=[1, -1, 4])
    g.nodes.append(
        gs.Node(
            op="Slice",
            name="slice_boxes_from_head",
            inputs=[
                head_bnc,
                const_i32("starts_boxes", [0, 0, 0]),
                const_i32("ends_boxes", [1, -1, 4]),
                const_i32("axes_boxes", [0, 1, 2]),
                const_i32("steps_boxes", [1, 1, 1]),
            ],
            outputs=[boxes],
        )
    )

    scores = gs.Variable("scores_bnc", dtype=np.float32, shape=[1, -1, 80])
    g.nodes.append(
        gs.Node(
            op="Slice",
            name="slice_scores_from_head",
            inputs=[
                head_bnc,
                const_i32("starts_scores", [0, 0, 4]),
                const_i32("ends_scores", [1, -1, 84]),
                const_i32("axes_scores", [0, 1, 2]),
                const_i32("steps_scores", [1, 1, 1]),
            ],
            outputs=[scores],
        )
    )

    return boxes, scores, f"{src}; sliced boxes [1,N,4] and scores [1,N,80]"


def insert_efficientnms_and_prune(
    g: gs.Graph,
    boxes: gs.Variable,
    scores: gs.Variable,
    keep_topk: int,
    score_thresh: float,
    iou_thresh: float,
    scores_are_probs: bool = True,
    box_coding_xyxy: bool = True,
) -> None:
    if boxes.shape is None or len(boxes.shape) != 3:
        boxes.shape = [1, -1, 4]
    if scores.shape is None or len(scores.shape) != 3:
        scores.shape = [1, -1, 80]

    out_count = gs.Variable("nms_num_dets", dtype=np.int32, shape=[1])
    out_boxes = gs.Variable(
        "nms_boxes", dtype=np.float32, shape=[1, keep_topk, 4]
    )
    out_scores = gs.Variable(
        "nms_scores", dtype=np.float32, shape=[1, keep_topk]
    )
    out_labels = gs.Variable(
        "nms_classes", dtype=np.int32, shape=[1, keep_topk]
    )

    g.nodes.append(
        gs.Node(
            op="EfficientNMS_TRT",
            name="node_efficientnms_trt",
            inputs=[boxes, scores],
            outputs=[out_count, out_boxes, out_scores, out_labels],
            attrs={
                "plugin_version": "1",
                "background_class": -1,
                "max_output_boxes": int(keep_topk),
                "score_threshold": float(score_thresh),
                "iou_threshold": float(iou_thresh),
                "score_activation": False if scores_are_probs else True,
                "box_coding": 0 if box_coding_xyxy else 1,
            },
        )
    )

    g.outputs = [out_count, out_boxes, out_scores, out_labels]

    g.nodes = [n for n in g.nodes if n.op not in {"NonMaxSuppression"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_onnx",
        required=True,
        help="Input ONNX (exported with nms=True)",
    )
    ap.add_argument(
        "--out",
        dest="out_onnx",
        required=True,
        help="Output ONNX (TRT-friendly)",
    )
    ap.add_argument("--keep_topk", type=int, default=100)
    ap.add_argument("--score_thresh", type=float, default=0.5)
    ap.add_argument("--iou_thresh", type=float, default=0.45)
    ap.add_argument(
        "--scores_are_logits",
        action="store_true",
        help="Set if the class scores are raw logits (rare for YOLO exports).",
    )
    ap.add_argument(
        "--boxes_are_xywh",
        action="store_true",
        help="Set if the 4 coords are xywh instead of xyxy (default xyxy).",
    )
    args = ap.parse_args()

    print(f"[load] {args.in_onnx}")
    g = gs.import_onnx(onnx.load(args.in_onnx))

    boxes, scores, dbg = try_find_boxes_scores_from_slices(g)
    print(f"[detect] {dbg}")
    if boxes is None or scores is None:
        boxes, scores, dbg2 = find_head_84_and_slice(g)
        print(f"[fallback] {dbg2}")

    insert_efficientnms_and_prune(
        g,
        boxes=boxes,
        scores=scores,
        keep_topk=args.keep_topk,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
        scores_are_probs=not args.scores_are_logits,
        box_coding_xyxy=not args.boxes_are_xywh,
    )

    g.cleanup().toposort()
    print(f"[save] {args.out_onnx}")
    onnx.save(gs.export_onnx(g), args.out_onnx)
    print("✅ Done")


if __name__ == "__main__":
    main()
