from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

    def __init__(self, names: Dict[int, str], width: int, height: int, santa_hat: Optional["SantaHat"] = None,
                 santa_hat_classes: Sequence[int] = ()):
        self.names = names
        self.width, self.height = width, height
        # Detections of these classes get a Santa hat instead of a box
        self.santa_hat = santa_hat
        self.santa_hat_classes = set(santa_hat_classes) if santa_hat is not None else set()
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
            if track.cls in self.santa_hat_classes:
                self.santa_hat(nv12, track.box)
                continue
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


class SantaHat:
    """
    Puts a Santa hat on top of a detection's box, blended straight into an NV12 frame with the image's alpha channel.

    The hat is scaled to a fraction of the box's shorter side (so an outstretched arm or a dog seen side-on does not make
    it huge), centred on the box, and its brim overlaps the top of the box a little so it sits on the head. Scaled copies are converted to YUV once and cached by width.
    """

    WIDTH_STEP = 16  # hat widths are rounded to this, so a moving box reuses a handful of cached sizes
    CACHE_SIZE = 64

    def __init__(self, image: Path, width: int, height: int, scale: float = 0.6, brim_overlap: float = 0.2):
        rgba = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"{image} must be a PNG with an alpha channel")
        ys, xs = np.nonzero(rgba[:, :, 3])
        self.rgba = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]  # trim the transparent border
        self.width, self.height = width, height
        self.scale, self.brim_overlap = scale, brim_overlap
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _sized(self, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Y plane, its alpha, interleaved UV plane and its alpha for a hat `w` pixels wide (float32)."""
        planes = self._cache.get(w)
        if planes is None:
            # Both sides multiples of 4, so the I420 chroma planes split cleanly and land on even frame pixels
            h = max(4, round(self.rgba.shape[0] * w / self.rgba.shape[1] / 4) * 4)
            rgba = cv2.resize(self.rgba, (w, h), interpolation=cv2.INTER_AREA)
            i420 = cv2.cvtColor(np.ascontiguousarray(rgba[:, :, :3]), cv2.COLOR_BGR2YUV_I420)
            u = i420[h : h + h // 4].reshape(h // 2, w // 2)
            v = i420[h + h // 4 :].reshape(h // 2, w // 2)
            alpha = rgba[:, :, 3].astype(np.float32) / 255.0
            planes = (
                i420[:h].astype(np.float32),
                alpha,
                np.dstack((u, v)).astype(np.float32),
                cv2.resize(alpha, (w // 2, h // 2), interpolation=cv2.INTER_AREA)[:, :, None],
            )
            if len(self._cache) >= self.CACHE_SIZE:
                self._cache.clear()
            self._cache[w] = planes
        return planes

    def __call__(self, nv12: np.ndarray, box: np.ndarray) -> None:
        """Draw the hat for an [x1, y1, x2, y2] box onto an (H * 3 / 2, W) NV12 array in place."""
        x1, y1, x2, y2 = (float(v) for v in box)
        w = min(round(min(x2 - x1, y2 - y1) * self.scale / self.WIDTH_STEP) * self.WIDTH_STEP, self.width // 4 * 4)
        if w < self.WIDTH_STEP:
            return
        hat_y, hat_a, hat_uv, hat_uv_a = self._sized(w)
        h = hat_y.shape[0]

        # Even coordinates keep the luma and chroma planes aligned
        left = round(((x1 + x2) / 2 - w / 2) / 2) * 2
        top = round((y1 + h * self.brim_overlap - h) / 2) * 2
        fx0, fy0 = max(left, 0), max(top, 0)
        fx1, fy1 = min(left + w, self.width), min(top + h, self.height)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        sx0, sy0 = fx0 - left, fy0 - top
        sx1, sy1 = sx0 + fx1 - fx0, sy0 + fy1 - fy0

        y_plane = nv12[: self.height]
        uv_plane = nv12[self.height :].reshape(self.height // 2, self.width // 2, 2)
        _blend(y_plane[fy0:fy1, fx0:fx1], hat_y[sy0:sy1, sx0:sx1], hat_a[sy0:sy1, sx0:sx1])
        _blend(
            uv_plane[fy0 // 2 : fy1 // 2, fx0 // 2 : fx1 // 2],
            hat_uv[sy0 // 2 : sy1 // 2, sx0 // 2 : sx1 // 2],
            hat_uv_a[sy0 // 2 : sy1 // 2, sx0 // 2 : sx1 // 2],
        )


def _blend(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> None:
    dst[...] = np.clip(dst * (1.0 - alpha) + src * alpha + 0.5, 0, 255).astype(np.uint8)
