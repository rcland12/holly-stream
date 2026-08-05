"""YOLO ONNX to TensorRT-compatible ONNX converter.

This module converts YOLO ONNX models to TensorRT-compatible ONNX models by
replacing the standard NMS operation with the EfficientNMS_TRT plugin. It
automatically detects and extracts bounding box and score tensors from various
YOLO export formats.

Typical usage:
    python yolo_efficient_nms_converter.py --in model.onnx --out model_trt.onnx
"""

import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs

DEFAULT_NUM_CLASSES = 80
DEFAULT_BOX_COORDS = 4
DEFAULT_HEAD_CHANNELS = DEFAULT_NUM_CLASSES + DEFAULT_BOX_COORDS  # 84


def build_tensor_name_map(graph: gs.Graph) -> Dict[str, gs.Variable]:
    """Build a mapping from tensor names to their corresponding gs.Variable objects.

    Creates a comprehensive map of all tensors in the graph, including inputs,
    outputs, and intermediate tensors from nodes.

    Args:
        graph: The ONNX graph to analyze.

    Returns:
        A dictionary mapping tensor names (str) to gs.Variable objects.
    """
    tensor_map = {tensor.name: tensor for tensor in graph.tensors().values()}

    for node in graph.nodes:
        for output in node.outputs:
            tensor_map.setdefault(output.name, output)

    for output in graph.outputs:
        tensor_map.setdefault(output.name, output)

    for input_tensor in graph.inputs:
        tensor_map.setdefault(input_tensor.name, input_tensor)

    return tensor_map


def _is_valid_3d_shape(shape: object) -> bool:
    """Check if a shape is a valid 3-dimensional shape.

    Args:
        shape: The shape to validate.

    Returns:
        True if the shape is a list or tuple with exactly 3 elements.
    """
    return isinstance(shape, (list, tuple)) and len(shape) == 3


def find_boxes_and_scores_from_slices(
    graph: gs.Graph,
) -> Tuple[Optional[gs.Variable], Optional[gs.Variable], str]:
    """Attempt to find box and score tensors from Slice operation outputs.

    Searches for typical YOLO decoded outputs where boxes and scores are
    produced by Slice operations with specific shapes:
        - boxes: Slice -> (1, N, 4)
        - scores: Slice -> (1, N, 80)

    Args:
        graph: The ONNX graph to search.

    Returns:
        A tuple containing:
            - boxes_tensor: The detected boxes tensor, or None if not found.
            - scores_tensor: The detected scores tensor, or None if not found.
            - debug_message: A string describing the detection result.
    """
    tensor_map = build_tensor_name_map(graph)
    boxes_tensor = None
    scores_tensor = None

    for node in graph.nodes:
        if node.op != "Slice":
            continue
        if not node.outputs or len(node.outputs) != 1:
            continue

        output_tensor = node.outputs[0]
        shape = getattr(output_tensor, "shape", None)

        if not _is_valid_3d_shape(shape):
            continue

        if shape[2] == DEFAULT_BOX_COORDS and boxes_tensor is None:
            boxes_tensor = output_tensor
        elif shape[2] == DEFAULT_NUM_CLASSES and scores_tensor is None:
            scores_tensor = output_tensor

    if boxes_tensor is not None and scores_tensor is not None:
        debug_message = (
            f"found Slice outputs: "
            f"boxes={boxes_tensor.name} {boxes_tensor.shape}, "
            f"scores={scores_tensor.name} {scores_tensor.shape}"
        )
        return boxes_tensor, scores_tensor, debug_message

    candidate_boxes = [
        tensor_map[name]
        for name in tensor_map
        if name.endswith("/Slice_output_0") or name.endswith("Slice_output_0")
    ]
    candidate_scores = [
        tensor_map[name]
        for name in tensor_map
        if name.endswith("/Slice_1_output_0")
        or name.endswith("Slice_1_output_0")
    ]

    if candidate_boxes and candidate_scores:
        return (
            candidate_boxes[0],
            candidate_scores[0],
            "fallback by common YOLO names",
        )

    return None, None, "no (1,N,4) and (1,N,80) Slice outputs found"


def find_head_tensor_and_create_slices(
    graph: gs.Graph,
) -> Tuple[gs.Variable, gs.Variable, str]:
    """Find the decoded head tensor and create box/score slices from it.

    This is a fallback method that searches for a decoded head tensor with 84
    channels (4 box coords + 80 class scores). The tensor can be in either:
        - BCN format: [1, 84, N] (requires transpose)
        - BNC format: [1, N, 84]

    Once found, the function creates Slice operations to extract:
        - boxes: [1, N, 4] from channels 0-3
        - scores: [1, N, 80] from channels 4-83

    Args:
        graph: The ONNX graph to modify.

    Returns:
        A tuple containing:
            - boxes_tensor: The created boxes tensor.
            - scores_tensor: The created scores tensor.
            - debug_message: A string describing the operation performed.

    Raises:
        RuntimeError: If no tensor with 84 channels can be found.
    """
    tensor_map = build_tensor_name_map(graph)

    bcn_candidates = []
    bnc_candidates = []

    for name, tensor in tensor_map.items():
        shape = getattr(tensor, "shape", None)
        if not _is_valid_3d_shape(shape):
            continue

        if shape[1] == DEFAULT_HEAD_CHANNELS:
            bcn_candidates.append((name, tensor, shape))
        if shape[2] == DEFAULT_HEAD_CHANNELS:
            bnc_candidates.append((name, tensor, shape))

    def _compute_preference_score(
        item: Tuple[str, gs.Variable, list],
    ) -> Tuple[int, str]:
        """Compute sorting key preferring Concat nodes and model subgraphs."""
        name, _, _ = item
        score = 0
        if "Concat" in name:
            score -= 2
        if "/model/" in name or "model." in name:
            score -= 1
        return (score, name)

    bcn_candidates.sort(key=_compute_preference_score)
    bnc_candidates.sort(key=_compute_preference_score)

    head_tensor_bnc = None
    source_description = ""

    if bcn_candidates:
        name, head_tensor_bcn, shape = bcn_candidates[0]
        head_tensor_bnc = gs.Variable(
            "head_bnc",
            dtype=np.float32,
            shape=[1, -1, DEFAULT_HEAD_CHANNELS],
        )
        graph.nodes.append(
            gs.Node(
                op="Transpose",
                name="node_head_transpose",
                inputs=[head_tensor_bcn],
                outputs=[head_tensor_bnc],
                attrs={"perm": [0, 2, 1]},
            )
        )
        source_description = (
            f"head BCN {name} {shape} -> transpose -> [1,N,84]"
        )

    elif bnc_candidates:
        name, head_tensor_bnc, shape = bnc_candidates[0]
        source_description = f"head BNC {name} {shape}"

    else:
        for node in graph.nodes:
            if node.op == "Concat" and node.outputs:
                concat_output_name = node.outputs[0].name
                head_tensor_bcn = tensor_map[concat_output_name]
                head_tensor_bnc = gs.Variable(
                    "head_bnc",
                    dtype=np.float32,
                    shape=[1, -1, DEFAULT_HEAD_CHANNELS],
                )
                graph.nodes.append(
                    gs.Node(
                        op="Transpose",
                        name="node_head_transpose_fb",
                        inputs=[head_tensor_bcn],
                        outputs=[head_tensor_bnc],
                        attrs={"perm": [0, 2, 1]},
                    )
                )
                source_description = f"fallback Concat {concat_output_name} -> transpose -> [1,N,84]"
                break
        else:
            raise RuntimeError(
                "Could not locate a decoded head with 84 channels."
            )

    boxes_tensor, scores_tensor = _create_box_score_slices(
        graph, head_tensor_bnc
    )

    debug_message = (
        f"{source_description}; sliced boxes [1,N,4] and scores [1,N,80]"
    )
    return boxes_tensor, scores_tensor, debug_message


def _create_int32_constant(name: str, values: list) -> gs.Constant:
    """Create an int32 constant tensor for Slice operation parameters.

    Args:
        name: The name for the constant tensor.
        values: The integer values for the constant.

    Returns:
        A gs.Constant object with int32 dtype.
    """
    return gs.Constant(name=name, values=np.array(values, dtype=np.int32))


def _create_box_score_slices(
    graph: gs.Graph,
    head_tensor: gs.Variable,
) -> Tuple[gs.Variable, gs.Variable]:
    """Create Slice operations to extract boxes and scores from the head tensor.

    Args:
        graph: The ONNX graph to modify.
        head_tensor: The head tensor in BNC format [1, N, 84].

    Returns:
        A tuple containing:
            - boxes_tensor: The sliced boxes tensor [1, N, 4].
            - scores_tensor: The sliced scores tensor [1, N, 80].
    """
    boxes_tensor = gs.Variable(
        "boxes_bnc",
        dtype=np.float32,
        shape=[1, -1, DEFAULT_BOX_COORDS],
    )
    graph.nodes.append(
        gs.Node(
            op="Slice",
            name="slice_boxes_from_head",
            inputs=[
                head_tensor,
                _create_int32_constant("starts_boxes", [0, 0, 0]),
                _create_int32_constant(
                    "ends_boxes", [1, -1, DEFAULT_BOX_COORDS]
                ),
                _create_int32_constant("axes_boxes", [0, 1, 2]),
                _create_int32_constant("steps_boxes", [1, 1, 1]),
            ],
            outputs=[boxes_tensor],
        )
    )

    scores_tensor = gs.Variable(
        "scores_bnc",
        dtype=np.float32,
        shape=[1, -1, DEFAULT_NUM_CLASSES],
    )
    graph.nodes.append(
        gs.Node(
            op="Slice",
            name="slice_scores_from_head",
            inputs=[
                head_tensor,
                _create_int32_constant(
                    "starts_scores", [0, 0, DEFAULT_BOX_COORDS]
                ),
                _create_int32_constant(
                    "ends_scores", [1, -1, DEFAULT_HEAD_CHANNELS]
                ),
                _create_int32_constant("axes_scores", [0, 1, 2]),
                _create_int32_constant("steps_scores", [1, 1, 1]),
            ],
            outputs=[scores_tensor],
        )
    )

    return boxes_tensor, scores_tensor


def insert_efficient_nms_and_prune(
    graph: gs.Graph,
    boxes_tensor: gs.Variable,
    scores_tensor: gs.Variable,
    max_detections: int,
    score_threshold: float,
    iou_threshold: float,
    scores_are_probabilities: bool = True,
    boxes_are_xyxy: bool = True,
) -> None:
    """Insert the EfficientNMS_TRT plugin and prune unnecessary nodes.

    Adds the TensorRT EfficientNMS plugin to the graph and removes any existing
    NonMaxSuppression operations. The graph outputs are replaced with the NMS
    plugin outputs.

    Args:
        graph: The ONNX graph to modify.
        boxes_tensor: The input boxes tensor [1, N, 4].
        scores_tensor: The input scores tensor [1, N, num_classes].
        max_detections: Maximum number of detections to keep.
        score_threshold: Minimum score threshold for detections.
        iou_threshold: IoU threshold for NMS.
        scores_are_probabilities: If True, scores are already probabilities.
            If False, sigmoid activation will be applied.
        boxes_are_xyxy: If True, boxes are in xyxy format.
            If False, boxes are in xywh format.
    """
    if boxes_tensor.shape is None or len(boxes_tensor.shape) != 3:
        boxes_tensor.shape = [1, -1, DEFAULT_BOX_COORDS]
    if scores_tensor.shape is None or len(scores_tensor.shape) != 3:
        scores_tensor.shape = [1, -1, DEFAULT_NUM_CLASSES]

    output_num_detections = gs.Variable(
        "nms_num_dets",
        dtype=np.int32,
        shape=[1],
    )
    output_boxes = gs.Variable(
        "nms_boxes",
        dtype=np.float32,
        shape=[1, max_detections, DEFAULT_BOX_COORDS],
    )
    output_scores = gs.Variable(
        "nms_scores",
        dtype=np.float32,
        shape=[1, max_detections],
    )
    output_class_ids = gs.Variable(
        "nms_classes",
        dtype=np.int32,
        shape=[1, max_detections],
    )

    graph.nodes.append(
        gs.Node(
            op="EfficientNMS_TRT",
            name="node_efficientnms_trt",
            inputs=[boxes_tensor, scores_tensor],
            outputs=[
                output_num_detections,
                output_boxes,
                output_scores,
                output_class_ids,
            ],
            attrs={
                "plugin_version": "1",
                "background_class": -1,
                "max_output_boxes": int(max_detections),
                "score_threshold": float(score_threshold),
                "iou_threshold": float(iou_threshold),
                "score_activation": not scores_are_probabilities,
                "box_coding": 0 if boxes_are_xyxy else 1,
            },
        )
    )

    graph.outputs = [
        output_num_detections,
        output_boxes,
        output_scores,
        output_class_ids,
    ]

    graph.nodes = [
        node for node in graph.nodes if node.op != "NonMaxSuppression"
    ]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert YOLO ONNX model to TensorRT-compatible format with EfficientNMS."
    )
    parser.add_argument(
        "--in",
        dest="input_onnx",
        required=True,
        help="Input ONNX model path (exported with nms=True).",
    )
    parser.add_argument(
        "--out",
        dest="output_onnx",
        required=True,
        help="Output ONNX model path (TRT-friendly).",
    )
    parser.add_argument(
        "--keep_topk",
        type=int,
        default=100,
        help="Maximum number of detections to keep (default: 100).",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="Score threshold for detections (default: 0.5).",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45).",
    )
    parser.add_argument(
        "--scores_are_logits",
        action="store_true",
        help="Set if class scores are raw logits (rare for YOLO exports).",
    )
    parser.add_argument(
        "--boxes_are_xywh",
        action="store_true",
        help="Set if box coordinates are xywh instead of xyxy (default: xyxy).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the YOLO to TensorRT converter."""
    args = parse_arguments()

    print(f"[load] {args.input_onnx}")
    graph = gs.import_onnx(onnx.load(args.input_onnx))

    boxes_tensor, scores_tensor, debug_message = (
        find_boxes_and_scores_from_slices(graph)
    )
    print(f"[detect] {debug_message}")

    if boxes_tensor is None or scores_tensor is None:
        boxes_tensor, scores_tensor, fallback_debug_message = (
            find_head_tensor_and_create_slices(graph)
        )
        print(f"[fallback] {fallback_debug_message}")

    insert_efficient_nms_and_prune(
        graph=graph,
        boxes_tensor=boxes_tensor,
        scores_tensor=scores_tensor,
        max_detections=args.keep_topk,
        score_threshold=args.score_thresh,
        iou_threshold=args.iou_thresh,
        scores_are_probabilities=not args.scores_are_logits,
        boxes_are_xyxy=not args.boxes_are_xywh,
    )

    graph.cleanup().toposort()
    print(f"[save] {args.output_onnx}")
    onnx.save(gs.export_onnx(graph), args.output_onnx)
    print("Done")


if __name__ == "__main__":
    main()
