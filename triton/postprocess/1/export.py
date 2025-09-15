from typing import List, Tuple

import torch
import torch.nn as nn


@torch.jit.script
def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from center-x, center-y, width, height to x1, y1, x2, y2.

    Args:
        x: Tensor of shape ``[..., 4]`` representing ``(cx, cy, w, h)`` in pixels.

    Returns:
        Tensor of shape ``[..., 4]`` representing ``(x1, y1, x2, y2)`` in pixels.
    """
    y = torch.empty_like(x)
    half_w = x[..., 2] * 0.5
    half_h = x[..., 3] * 0.5
    y[..., 0] = x[..., 0] - half_w
    y[..., 1] = x[..., 1] - half_h
    y[..., 2] = x[..., 0] + half_w
    y[..., 3] = x[..., 1] + half_h
    return y


@torch.jit.script
def nms_ts(
    boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float, pre_topk: int
) -> torch.Tensor:
    """
    Non-maximum suppression implemented for TorchScript.

    Args:
        boxes: Tensor of shape ``[N, 4]`` in ``(x1, y1, x2, y2)`` format.
        scores: Tensor of shape ``[N]`` with confidence scores.
        iou_thres: IoU threshold for suppression.
        pre_topk: Optional pre-topk to limit candidates before NMS. Use ``<=0`` to disable.

    Returns:
        Tensor of dtype ``long`` with indices of kept boxes, sorted by score.
    """
    N = scores.numel()
    if N == 0:
        return scores.new_empty((0,), dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    if pre_topk > 0 and order.numel() > pre_topk:
        order = order[:pre_topk]

    keep_list = torch.jit.annotate(List[int], [])

    while order.numel() > 0:
        i = int(order[0].item())
        keep_list.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]

        x1 = torch.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = torch.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = torch.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = torch.minimum(boxes[i, 3], boxes[rest, 3])

        inter_w = torch.clamp(x2 - x1, min=0.0)
        inter_h = torch.clamp(y2 - y1, min=0.0)
        inter = inter_w * inter_h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (
            boxes[rest, 3] - boxes[rest, 1]
        )
        union = area_i + area_r - inter + 1e-9
        ious = inter / union

        mask = ious <= iou_thres
        order = rest[mask]

        if len(keep_list) >= 300:
            break

    return torch.tensor(keep_list, dtype=torch.long, device=boxes.device)


@torch.jit.script
def clamp_boxes(xyxy: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """
    Clamp box coordinates to image boundaries.

    Args:
        xyxy: Tensor of shape ``[N, 4]`` in ``(x1, y1, x2, y2)`` format.
        w: Image width.
        h: Image height.

    Returns:
        Tensor of shape ``[N, 4]`` with coordinates clamped to ``[0, w]`` and ``[0, h]``.
    """
    xyxy[:, 0].clamp_(0, float(w))
    xyxy[:, 1].clamp_(0, float(h))
    xyxy[:, 2].clamp_(0, float(w))
    xyxy[:, 3].clamp_(0, float(h))
    return xyxy


@torch.jit.script
def postprocess_batch(
    det: torch.Tensor,
    img_w: int,
    img_h: int,
    model_w: int,
    model_h: int,
    conf_thres: float,
    iou_thres: float,
    max_dets: int,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert model detections to clamped xyxy, scores, and class ids with NMS.

    Args:
        det: Tensor of shape ``[B, C, K]`` where first 4 are boxes and next ``num_classes`` are class logits.
        img_w: Original image width.
        img_h: Original image height.
        model_w: Model input width.
        model_h: Model input height.
        conf_thres: Confidence threshold.
        iou_thres: IoU threshold for NMS.
        max_dets: Maximum detections per image to keep.
        num_classes: Number of classes.

    Returns:
        A tuple ``(boxes_out, count_out)`` where:
          - ``boxes_out`` is ``[B, max_dets, 6]`` as ``(x1,y1,x2,y2,score,class_id)``.
          - ``count_out`` is ``[B, 1]`` containing counts per batch item.
    """
    B = det.shape[0]
    nc = num_classes

    det = det.transpose(1, 2)
    boxes_cxcywh = det[..., 0:4]
    cls_logits = det[..., 4 : 4 + nc]

    conf_vals, cls_idx = torch.max(cls_logits, dim=-1)
    keep = conf_vals > conf_thres

    mw = float(model_w)
    mh = float(model_h)
    iw = float(img_w)
    ih = float(img_h)
    gain = mh / ih if (mh / ih) < (mw / iw) else mw / iw
    pad_x = (mw - iw * gain) * 0.5
    pad_y = (mh - ih * gain) * 0.5

    boxes_xyxy = xywh2xyxy(boxes_cxcywh)
    boxes_xyxy[..., 0] -= pad_x
    boxes_xyxy[..., 2] -= pad_x
    boxes_xyxy[..., 1] -= pad_y
    boxes_xyxy[..., 3] -= pad_y
    boxes_xyxy /= gain

    boxes_out = det.new_zeros((B, max_dets, 6), dtype=torch.float32)
    count_out = torch.zeros((B, 1), dtype=torch.int32, device=det.device)

    for b in range(B):
        m = keep[b].nonzero().view(-1)
        if m.numel() == 0:
            continue

        boxes_b = boxes_xyxy[b][m]
        scores = conf_vals[b][m]
        clsb = cls_idx[b][m].to(torch.float32)

        offsets = clsb.view(-1, 1) * 7680.0
        nms_boxes = boxes_b + offsets

        pre_topk = 30000
        keep_idx = nms_ts(nms_boxes, scores, iou_thres, pre_topk)
        if keep_idx.numel() > max_dets:
            keep_idx = keep_idx[:max_dets]

        nb = int(keep_idx.numel())
        if nb > 0:
            out = torch.empty(
                (nb, 6), dtype=torch.float32, device=boxes_b.device
            )
            sel = keep_idx
            out[:, 0:4] = clamp_boxes(boxes_b[sel], img_w, img_h)
            out[:, 4] = scores[sel]
            out[:, 5] = clsb[sel]
            boxes_out[b, 0:nb, :] = out
            count_out[b, 0] = nb

    return boxes_out, count_out


class PostprocessTS(nn.Module):
    """
    TorchScript-ready postprocessing for object detection outputs.

    Args:
        img_w: Original image width.
        img_h: Original image height.
        model_w: Model input width.
        model_h: Model input height.
        conf_thres: Confidence threshold.
        iou_thres: IoU threshold for NMS.
        max_dets: Maximum detections per image to keep.
        num_classes: Number of classes.
    """

    def __init__(
        self,
        img_w: int = 1280,
        img_h: int = 720,
        model_w: int = 640,
        model_h: int = 640,
        conf_thres: float = 0.3,
        iou_thres: float = 0.25,
        max_dets: int = 300,
        num_classes: int = 80,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "cfg",
            torch.tensor(
                [
                    float(img_w),
                    float(img_h),
                    float(model_w),
                    float(model_h),
                    float(conf_thres),
                    float(iou_thres),
                    float(max_dets),
                    float(num_classes),
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, det: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply postprocessing to a batch of raw model outputs.

        Args:
            det: Tensor of shape ``[B, C, K]`` where first 4 are boxes and next ``num_classes`` are class logits.

        Returns:
            Tuple of ``(boxes_out, count_out)`` as described in :func:`postprocess_batch`.
        """
        img_w = int(self.cfg[0].item())
        img_h = int(self.cfg[1].item())
        model_w = int(self.cfg[2].item())
        model_h = int(self.cfg[3].item())
        conf_thres = float(self.cfg[4].item())
        iou_thres = float(self.cfg[5].item())
        max_dets = int(self.cfg[6].item())
        num_classes = int(self.cfg[7].item())
        return postprocess_batch(
            det,
            img_w,
            img_h,
            model_w,
            model_h,
            conf_thres,
            iou_thres,
            max_dets,
            num_classes,
        )


def build_model(device: str = "cuda") -> torch.jit.ScriptModule:
    """
    Script and return the postprocessing model.

    Args:
        device: Target device string, e.g., ``"cuda"`` or ``"cpu"``.

    Returns:
        A ``torch.jit.ScriptModule`` of ``PostprocessTS`` placed on the requested device.
    """
    m = PostprocessTS().eval()
    m = m.to(device)
    scripted = torch.jit.script(m)
    return scripted


def make_example_input(device: str = "cuda") -> torch.Tensor:
    """
    Create an example raw detection tensor.

    Args:
        device: Target device string for the tensor allocation.

    Returns:
        Tensor with shape ``(1, 84, 8400)`` and dtype ``float32`` on the requested device.
    """
    return torch.zeros((1, 84, 8400), device=device, dtype=torch.float32)


def script_and_save(
    out_path: str, device: str = "cuda", verify: bool = False
) -> None:
    """
    Script the postprocess model, optionally verify, and save to disk.

    Args:
        out_path: Destination path for the serialized TorchScript file.
        device: Target device string for example input generation.
        verify: If ``True``, prints output shapes, dtypes, and devices.
    """
    scripted = build_model(device=device)
    with torch.inference_mode():
        dummy = make_example_input(device=device)
        out = scripted(dummy)
        if verify:
            if isinstance(out, tuple):
                shapes = [tuple(t.shape) for t in out]
                dtypes = [str(t.dtype) for t in out]
                devices = [str(t.device) for t in out]
                print(f"[verify] outputs.shapes = {shapes}")
                print(f"[verify] outputs.dtypes = {dtypes}")
                print(f"[verify] outputs.devices = {devices}")
            else:
                print(
                    f"[verify] output.shape = {tuple(out.shape)}, dtype = {out.dtype}, device = {out.device}"
                )

    scripted.save(out_path)
    print(f"[done] saved TorchScript to: {out_path}")


def main() -> None:
    """
    Entrypoint to script and save the model to ``./model.pt`` with verification.
    """
    script_and_save(out_path="./model.pt", verify=True)


if __name__ == "__main__":
    main()
