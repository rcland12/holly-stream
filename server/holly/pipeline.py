"""
One camera's pipeline, run in its own worker process:

    RTSP from the ingest -> NVDEC -> GPU preprocess -> TensorRT -> boxes drawn on NV12 -> NVENC -> RTMP to the ingest

Audio packets are copied through untouched and every frame keeps its source timestamp, so audio and video stay
in sync.
"""

import logging
import time
from fractions import Fraction
from typing import Callable, Dict, List, Optional

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from holly.config import Config
from holly.engine import Engine
from holly.overlay import BoxSmoother, NV12Annotator

log = logging.getLogger(__name__)

FPS_MEASURE_FRAMES = 6


class Preprocessor:
    """
    Turns a decoded NV12 frame into the model's letterboxed RGB input, on the GPU.

    RGB is computed at the chroma plane's resolution (half the frame in each direction), so a 1280x960 camera
    frame becomes exactly 640x480 with no resize step, which suits a 480x640 model.
    """

    def __init__(self, width: int, height: int, model_h: int, model_w: int, dtype: torch.dtype):
        self.width, self.height = width, height
        half_h, half_w = height // 2, width // 2
        self.scale = min(model_h / half_h, model_w / half_w)
        self.resized_h, self.resized_w = round(half_h * self.scale), round(half_w * self.scale)
        self.pad_top = (model_h - self.resized_h) // 2
        self.pad_left = (model_w - self.resized_w) // 2
        # Pinned host memory makes the upload a single DMA copy
        self.host = torch.empty(height * 3 // 2, width, dtype=torch.uint8).pin_memory()
        self.canvas = torch.full((1, 3, model_h, model_w), 114 / 255, dtype=dtype, device="cuda")

    def __call__(self, nv12: np.ndarray) -> torch.Tensor:
        self.host.numpy()[:] = nv12
        frame = self.host.to("cuda", non_blocking=True)
        y = frame[: self.height].view(1, 1, self.height, self.width).float()
        y = F.avg_pool2d(y, 2)[0, 0]
        uv = frame[self.height :].view(self.height // 2, self.width // 2, 2).float() - 128.0
        u, v = uv[..., 0], uv[..., 1]

        # BT.709 limited range, which is what the Pi's encoder signals
        y = (y - 16.0) * 1.164
        rgb = torch.stack((y + 1.793 * v, y - 0.213 * u - 0.533 * v, y + 2.112 * u)).clamp_(0, 255).div_(255.0)

        if (self.resized_h, self.resized_w) != rgb.shape[1:]:
            rgb = F.interpolate(rgb[None], size=(self.resized_h, self.resized_w), mode="bilinear", align_corners=False)[0]
        self.canvas[0, :, self.pad_top : self.pad_top + self.resized_h, self.pad_left : self.pad_left + self.resized_w] = rgb
        return self.canvas

    def to_frame_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Map [x1, y1, x2, y2] boxes from model input pixels back to full-resolution frame pixels."""
        boxes = boxes.clone()
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - self.pad_left) / self.scale * 2).clamp_(0, self.width - 1)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - self.pad_top) / self.scale * 2).clamp_(0, self.height - 1)
        return boxes


class Stats:
    """Averages per-frame timings and hands them to `report` every `interval` seconds."""

    def __init__(self, interval: float, report: Callable[[Dict[str, float]], None]):
        self.interval, self.report = interval, report
        self._reset()

    def _reset(self) -> None:
        self.start = time.monotonic()
        self.frames = 0
        self.infer_s = self.process_s = 0.0
        self.boxes = 0

    def add(self, infer_s: float, process_s: float, boxes: int) -> None:
        self.frames += 1
        self.infer_s += infer_s
        self.process_s += process_s
        self.boxes += boxes
        elapsed = time.monotonic() - self.start
        if elapsed >= self.interval:
            self.report({
                "fps": round(self.frames / elapsed, 1),
                "inference_ms": round(self.infer_s / self.frames * 1000, 1),
                "frame_ms": round(self.process_s / self.frames * 1000, 1),
                "boxes_per_frame": round(self.boxes / self.frames, 2),
            })
            self._reset()


class Snapshots:
    """Saves a clean (unannotated) frame every `interval` seconds as <camera>_<timestamp>.jpg."""

    def __init__(self, config: Config, name: str):
        self.directory, self.interval = config.snapshot_dir, config.snapshot_interval
        self.prefix = name.replace("/", "_")
        self.next_save = 0.0
        self.directory.mkdir(parents=True, exist_ok=True)

    def maybe_save(self, nv12: np.ndarray) -> None:
        now = time.time()
        if now < self.next_save:
            return
        self.next_save = now + self.interval
        path = self.directory / f"{self.prefix}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12), [cv2.IMWRITE_JPEG_QUALITY, 95])


class Session:
    """
    Annotation, encoding and publishing for one connected camera.

    Created from the first decoded frames rather than stream probing: a stream joined before its next keyframe
    probes as 0x0, and the probed frame rate is sometimes doubled. The frame rate is measured from the first
    frames' timestamps instead, because the encoder's rate control depends on it.
    """

    def __init__(self, config: Config, engine: Engine, name: str, rotate_180: bool, frames: List[av.VideoFrame],
                 video_in: av.video.stream.VideoStream, audio_in: Optional[av.audio.stream.AudioStream],
                 report: Callable[[Dict], None]):
        self.config, self.engine, self.rotate_180 = config, engine, rotate_180
        self.video_in, self.audio_in = video_in, audio_in
        self.width, self.height = frames[0].width, frames[0].height
        fps = _measure_fps(frames, video_in.time_base)
        log.info(
            "Camera %s: %dx%d @ %.2f fps, audio=%s, rotate_180=%s",
            name, self.width, self.height, float(fps), audio_in.codec_context.name if audio_in else "none", rotate_180,
        )

        # The encoder stream is added directly (not copied from a template stream): PyAV opens a fresh encoder for
        # template streams, which would start a second NVENC session with a stream header that does not match.
        # Audio is copied from the input, which only opens a decoder.
        self.output_url = config.output_url.format(name=name)
        self.output = av.open(self.output_url, "w", format="flv", timeout=10.0)
        self.video_out = self.output.add_stream("h264_nvenc", rate=fps)
        self.video_out.width, self.video_out.height, self.video_out.pix_fmt = self.width, self.height, "nv12"
        self.video_out.time_base = video_in.time_base
        self.video_out.codec_context.time_base = video_in.time_base
        self.video_out.options = {
            "preset": config.nvenc_preset,
            "tune": "hq",
            "rc": "vbr",
            "b": f"{config.bitrate_kbps}k",
            "maxrate": f"{config.bitrate_kbps * 3 // 2}k",
            "bufsize": f"{config.bitrate_kbps * 2}k",
            # Keyframes are forced by timestamp in process_video; this is only an upper bound
            "g": str(round(float(fps) * config.keyint_seconds * 2)),
            "forced-idr": "1",
            "bf": "0",
            "profile": "high",
            "spatial-aq": "1",
        }
        self.audio_out = self.output.add_stream(template=audio_in) if audio_in else None
        log.info("Publishing %s to %s", name, _redact(self.output_url))

        model_h, model_w = engine.input_size
        self.preprocess = Preprocessor(self.width, self.height, model_h, model_w, engine.input_dtype)
        self.annotate = NV12Annotator(engine.names, self.width, self.height)
        self.smoother = BoxSmoother(smoothing=config.smoothing)
        self.class_filter = torch.tensor(config.classes, device="cuda") if config.classes else None
        self.snapshots = Snapshots(config, name) if config.snapshot_interval > 0 else None
        self.stats = Stats(config.stats_interval, lambda stats: report({**stats, "resolution": f"{self.width}x{self.height}"}))

        # Output timestamps start at zero from the first frame; audio is shifted by the same amount
        self.start_pts = frames[0].pts
        self.audio_offset = round(self.start_pts * video_in.time_base / audio_in.time_base) if audio_in else 0
        self.keyint_pts = round(config.keyint_seconds / video_in.time_base)
        self.next_keyframe_pts = 0

    def process_video(self, frame: av.VideoFrame) -> None:
        if (frame.width, frame.height) != (self.width, self.height):
            raise ValueError(f"Camera resolution changed to {frame.width}x{frame.height}")
        t0 = time.monotonic()

        nv12 = frame.to_ndarray()
        if self.rotate_180:
            nv12 = _rotate_nv12(nv12, self.height)

        images = self.preprocess(nv12)
        t_infer = time.monotonic()
        detections = self.engine(images)
        infer_s = time.monotonic() - t_infer

        keep = detections[:, 4] >= self.config.confidence
        if self.class_filter is not None:
            keep &= torch.isin(detections[:, 5].long(), self.class_filter)
        detections = detections[keep]
        detections[:, :4] = self.preprocess.to_frame_boxes(detections[:, :4])
        detections = detections.cpu().numpy()  # one device-to-host copy per frame

        if self.snapshots is not None:
            self.snapshots.maybe_save(nv12)

        tracks = self.smoother.update(detections[:, :4], detections[:, 4], detections[:, 5].astype(int))
        self.annotate(nv12, tracks)

        out_frame = av.VideoFrame.from_ndarray(nv12, format="nv12")
        out_frame.pts = frame.pts - self.start_pts
        out_frame.time_base = self.video_in.time_base
        # Keyframes on a fixed clock line up with the HLS fragments downstream at any frame rate
        if out_frame.pts >= self.next_keyframe_pts:
            out_frame.pict_type = av.video.frame.PictureType.I
            self.next_keyframe_pts = out_frame.pts - out_frame.pts % self.keyint_pts + self.keyint_pts
        for packet in self.video_out.encode(out_frame):
            self.output.mux(packet)

        self.stats.add(infer_s, time.monotonic() - t0, len(tracks))

    def process_audio(self, packet: av.Packet) -> None:
        if self.audio_out is None or packet.pts is None or packet.pts - self.audio_offset < 0:
            return
        packet.pts -= self.audio_offset
        packet.dts = packet.pts if packet.dts is None else packet.dts - self.audio_offset
        packet.stream = self.audio_out
        self.output.mux(packet)

    def close(self) -> None:
        # The ingest may already be gone, so closing is best effort
        try:
            for packet in self.video_out.encode(None):
                self.output.mux(packet)
            self.output.close()
        except Exception as e:
            log.debug("Error closing output: %s", e)


def run_camera(config: Config, engine: Engine, name: str, rotate_180: bool, report: Callable[[Dict], None]) -> None:
    """Process one camera from the ingest until its stream ends."""
    url = f"{config.ingest_rtsp}/{name}"
    source = av.open(url, options={"rtsp_transport": "tcp", "fflags": "nobuffer"}, timeout=(10.0, 10.0))
    session: Optional[Session] = None
    try:
        video_in = source.streams.video[0]
        audio_in = source.streams.audio[0] if source.streams.audio else None
        decoder = av.CodecContext.create("h264_cuvid", "r")
        if video_in.codec_context.extradata:
            decoder.extradata = video_in.codec_context.extradata

        pending: List[av.VideoFrame] = []  # frames held back until the frame rate is measured
        for packet in source.demux([video_in] + ([audio_in] if audio_in else [])):
            if packet.dts is None and packet.pts is None:
                continue
            if packet.stream is audio_in:
                if session is not None:
                    session.process_audio(packet)
                continue
            for frame in decoder.decode(packet):
                if frame.pts is None:
                    continue
                if session is None:
                    if pending and frame.pts <= pending[-1].pts:
                        pending.clear()  # out of order while joining; start over
                    pending.append(frame)
                    if len(pending) < FPS_MEASURE_FRAMES:
                        continue
                    session = Session(config, engine, name, rotate_180, pending, video_in, audio_in, report)
                    for held in pending:
                        session.process_video(held)
                    pending.clear()
                    continue
                session.process_video(frame)
        log.info("Camera %s stream ended", name)
    finally:
        if session is not None:
            session.close()
        try:
            source.close()
        except Exception as e:
            log.debug("Error closing input: %s", e)


def _measure_fps(frames: List[av.VideoFrame], time_base: Fraction) -> Fraction:
    """Average frame rate over consecutive frames, snapped to the nearest common rate when within 10%."""
    measured = (len(frames) - 1) / float((frames[-1].pts - frames[0].pts) * time_base)
    common = [Fraction(15), Fraction(24), Fraction(25), Fraction(30), Fraction(50), Fraction(60)]
    nearest = min(common, key=lambda rate: abs(float(rate) - measured))
    return nearest if abs(float(nearest) - measured) <= 0.1 * float(nearest) else Fraction(round(measured))


def _redact(url: str) -> str:
    """Hide the last path segment (an RTMP stream key) in logs."""
    head, _, key = url.rpartition("/")
    return f"{head}/{'*' * min(len(key), 8)}" if key else url


def _rotate_nv12(nv12: np.ndarray, height: int) -> np.ndarray:
    y = nv12[:height, ::-1][::-1]
    uv = nv12[height:].reshape(height // 2, -1, 2)[::-1, ::-1]
    return np.ascontiguousarray(np.vstack((y, uv.reshape(height // 2, -1))))
