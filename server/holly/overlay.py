from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class Track:
    box: np.ndarray  # [x1, y1, x2, y2] in frame pixels
    score: float
    cls: int
    hits: int = 1
    misses: int = 0


@dataclass
class BoxSmoother:
    """
    Steadies per-frame detections before they are drawn.

    Detections are matched to the previous frame's boxes by IoU within the same class. Matched boxes are
    blended toward the new position, which removes frame-to-frame jitter; a box that is missed for a few
    frames stays on screen instead of flickering; and a new box only appears once it has been seen twice,
    which hides one-frame false positives.
    """

    smoothing: float = 0.5  # weight of the previous position, 0 = no smoothing
    max_misses: int = 4
    min_hits: int = 2
    iou_threshold: float = 0.3
    tracks: List[Track] = field(default_factory=list)

    def update(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> List[Track]:
        unmatched = set(range(len(boxes)))
        for track in sorted(self.tracks, key=lambda t: -t.score):
            best, best_iou = -1, self.iou_threshold
            for i in unmatched:
                if classes[i] != track.cls:
                    continue
                iou = _iou(track.box, boxes[i])
                if iou > best_iou:
                    best, best_iou = i, iou
            if best >= 0:
                unmatched.discard(best)
                track.box = self.smoothing * track.box + (1 - self.smoothing) * boxes[best]
                track.score = float(scores[best])
                track.hits += 1
                track.misses = 0
            else:
                track.misses += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        self.tracks += [Track(boxes[i].astype(np.float32), float(scores[i]), int(classes[i])) for i in unmatched]
        return [t for t in self.tracks if t.hits >= self.min_hits]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class NV12Annotator:
    """
    Draws boxes and labels straight onto an NV12 frame (a Y plane followed by an interleaved UV plane at half
    resolution), so the frame never has to be converted to RGB and back before encoding.
    """

    def __init__(self, names: Dict[int, str], width: int, height: int):
        self.names = names
        self.width, self.height = width, height
        # Scale line widths and text with the frame so labels stay readable at any resolution
        self.thickness = max(2, round(height / 360))
        self.font_scale = height / 1200
        rng = np.random.default_rng(7)
        self.colors = {cls: self._bgr_to_yuv(tuple(int(c) for c in rng.integers(60, 256, 3))) for cls in names}

    @staticmethod
    def _bgr_to_yuv(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
        i420 = cv2.cvtColor(np.full((2, 2, 3), bgr, dtype=np.uint8), cv2.COLOR_BGR2YUV_I420)
        return int(i420[0, 0]), int(i420[2, 0]), int(i420[2, 1])

    def __call__(self, nv12: np.ndarray, tracks: List[Track]) -> None:
        """Annotate an (H * 3 / 2, W) NV12 array in place."""
        y_plane = nv12[: self.height]
        uv_plane = nv12[self.height :].reshape(self.height // 2, self.width // 2, 2)

        def rect(p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], thickness: int) -> None:
            cv2.rectangle(y_plane, p1, p2, color[0], thickness)
            uv_thickness = thickness if thickness < 0 else max(1, thickness // 2)
            cv2.rectangle(uv_plane, (p1[0] // 2, p1[1] // 2), (p2[0] // 2, p2[1] // 2), color[1:], uv_thickness)

        for track in tracks:
            x1, y1, x2, y2 = (int(v) for v in track.box)
            color = self.colors[track.cls]
            rect((x1, y1), (x2, y2), color, self.thickness)

            text = f"{self.names[track.cls]} {track.score:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            pad = max(2, th // 4)
            top = y1 - th - baseline - 2 * pad
            if top < 0:
                top = y1
            rect((x1, top), (x1 + tw + 2 * pad, top + th + baseline + 2 * pad), color, -1)
            cv2.putText(
                y_plane,
                text,
                (x1 + pad, top + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                16 if color[0] > 128 else 235,
                max(1, self.thickness // 2),
                cv2.LINE_AA,
            )
