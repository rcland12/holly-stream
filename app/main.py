import os
import queue
import signal
import subprocess
import threading
import time
from ast import literal_eval
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
from urllib.parse import urlparse

import cv2
import imutils
import numpy as np


class EnvArgumentParser:
    """
    Environment-backed argument parser with type casting.
    """

    def __init__(self) -> None:
        """
        Initialize an empty argument dictionary.
        """
        self.dict: Dict[str, Any] = {}

    class _define_dict(dict):
        """
        Dictionary subclass allowing attribute-style access.
        """

        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def add_arg(
        self, variable: str, default: Any = None, d_type: Type = str
    ) -> None:
        """
        Register a variable to parse from the environment.

        Args:
            variable: Environment variable name.
            default: Fallback value if the environment variable is missing.
            d_type: Callable or type used to cast the environment value.

        Raises:
            TypeError: If default cannot be cast to the provided type.
            ValueError: If environment value cannot be parsed to the provided type.
            SyntaxError: If literal evaluation fails for structured types.
        """
        env = os.environ.get(variable)
        if env is None:
            if isinstance(default, d_type):
                value = default
            else:
                raise TypeError(
                    f"The default value for {variable} cannot be cast to the data type provided."
                )
        else:
            if callable(d_type):
                value = self._cast_type(env, d_type)
            else:
                value = env
        self.dict[variable] = value

    @staticmethod
    def _cast_type(arg: str, d_type: Type) -> Any:
        """
        Cast a string argument to a given data type.

        Args:
            arg: String value to cast.
            d_type: Target type or constructor.

        Returns:
            The cast value.

        Raises:
            TypeError: If the cast result is not an instance of the requested type.
            ValueError: If casting fails.
            SyntaxError: If literal evaluation fails due to syntax.
        """
        if d_type in [list, tuple, bool, dict]:
            try:
                cast_value = literal_eval(arg)
                if not isinstance(cast_value, d_type):
                    raise TypeError(
                        f"The value cast type ({d_type}) does not match the value given for {arg}"
                    )
            except ValueError as e:
                raise ValueError(
                    f"Argument {arg} does not match given data type or is not supported:",
                    str(e),
                )
            except SyntaxError as e:
                raise SyntaxError(
                    f"Check the types entered for argument {arg}:", str(e)
                )
        else:
            try:
                cast_value = d_type(arg)
            except ValueError as e:
                raise ValueError(
                    f"Argument {arg} does not match given data type or is not supported:",
                    str(e),
                )
            except SyntaxError as e:
                raise SyntaxError(
                    f"Check the types entered for argument {arg}:", str(e)
                )
        return cast_value

    def parse_args(self) -> "_define_dict":
        """
        Return parsed arguments with attribute access.

        Returns:
            A dictionary-like object supporting attribute-style access.
        """
        return self._define_dict(self.dict)


class LatestFrame:
    """
    Thread-safe container for the latest video frame.
    """

    def __init__(self) -> None:
        """
        Initialize the container.
        """
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: float = 0.0

    def set(self, frame: np.ndarray) -> None:
        """
        Store a frame and timestamp.

        Args:
            frame: Frame array in BGR format.
        """
        with self._lock:
            self._frame = frame
            self._ts = time.time()

    def get(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Retrieve a copy of the latest frame and its timestamp.

        Returns:
            A tuple with (frame copy or None, timestamp seconds).
        """
        with self._lock:
            return (
                None if self._frame is None else self._frame.copy(),
                self._ts,
            )


class AtomicDetections:
    """
    Thread-safe container for detections.
    """

    def __init__(self) -> None:
        """
        Initialize empty detections.
        """
        self._lock = threading.Lock()
        self._bboxes: List[List[float]] = []
        self._confs: List[float] = []
        self._indexes: List[int] = []
        self._ts: float = 0.0

    def set(self, b: List[List[float]], c: List[float], i: List[int]) -> None:
        """
        Store detection results and timestamp.

        Args:
            b: Bounding boxes [[x1,y1,x2,y2], ...].
            c: Confidence scores.
            i: Class indices.
        """
        with self._lock:
            self._bboxes, self._confs, self._indexes = b, c, i
            self._ts = time.time()

    def get(self) -> Tuple[List[List[float]], List[float], List[int], float]:
        """
        Retrieve current detections.

        Returns:
            A tuple of (bboxes, confs, indexes, timestamp seconds).
        """
        with self._lock:
            return (self._bboxes, self._confs, self._indexes, self._ts)


class TritonClient:
    """
    Minimal Triton client wrapper supporting HTTP and gRPC.
    """

    def __init__(
        self,
        url: str,
        model: str,
        requested_outputs: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Create a Triton client.

        Args:
            url: Server URL in form http://host:port or grpc://host:port.
            model: Model name.
            requested_outputs: Optional list of output tensor names.

        Raises:
            RuntimeError: If an unsupported scheme is provided.
        """
        self.model_name: str = model
        parsed = urlparse(url)
        self.scheme: str = parsed.scheme.lower()
        netloc = parsed.netloc
        if self.scheme not in ("grpc", "http"):
            raise RuntimeError("Use http://host:port or grpc://host:port")

        if self.scheme == "grpc":
            from tritonclient.grpc import (
                InferenceServerClient,
                InferInput,
                InferRequestedOutput,
            )

            self._InferInput = InferInput
            self._InferRequestedOutput = InferRequestedOutput
            self.client = InferenceServerClient(netloc, verbose=False)
            self.metadata = self.client.get_model_metadata(
                self.model_name, as_json=True
            )
            self.config = self.client.get_model_config(
                self.model_name, as_json=True
            )["config"]
        else:
            from tritonclient.http import (
                InferenceServerClient,
                InferInput,
                InferRequestedOutput,
            )

            self._InferInput = InferInput
            self._InferRequestedOutput = InferRequestedOutput
            self.client = InferenceServerClient(netloc, verbose=False)
            self.metadata = self.client.get_model_metadata(self.model_name)
            self.config = self.client.get_model_config(self.model_name)

        self.labels: Optional[List[str]] = self._get_labels()
        self._input_schemas: List[Tuple[str, List[int], str]] = [
            (i["name"], [int(s) for s in i["shape"]], i["datatype"])
            for i in self.metadata["inputs"]
        ]
        self.output_names: List[str] = (
            [o["name"] for o in self.metadata["outputs"]]
            if requested_outputs is None
            else list(requested_outputs)
        )

        self._requested_outputs_objs: List[Any] = []
        for out_name in self.output_names:
            if self.scheme == "http":
                self._requested_outputs_objs.append(
                    self._InferRequestedOutput(out_name, binary_data=True)
                )
            else:
                self._requested_outputs_objs.append(
                    self._InferRequestedOutput(out_name)
                )

    def _triton_dtype_to_numpy(self, dtype: str) -> Any:
        """
        Map Triton dtype string to NumPy dtype.

        Args:
            dtype: Triton datatype string.

        Returns:
            Corresponding NumPy dtype object.

        Raises:
            KeyError: If dtype is unsupported.
        """
        import numpy as np

        m = {
            "UINT8": np.uint8,
            "INT8": np.int8,
            "INT16": np.int16,
            "INT32": np.int32,
            "INT64": np.int64,
            "FP16": np.float16,
            "FP32": np.float32,
            "FP64": np.float64,
            "BOOL": np.bool_,
        }
        return m[dtype]

    def _pack_inputs(self, *args: Any) -> List[Any]:
        """
        Create Triton input objects from numpy-like arrays.

        Args:
            *args: Arrays corresponding to model inputs.

        Returns:
            A list of Triton input objects.
        """
        placeholders: List[Any] = []
        for (name, _, dtype), value in zip(self._input_schemas, args):
            arr = value if isinstance(value, np.ndarray) else np.asarray(value)
            npdtype = self._triton_dtype_to_numpy(dtype)
            if arr.dtype != npdtype:
                arr = arr.astype(npdtype, copy=False)
            arr = np.ascontiguousarray(arr)
            inp = self._InferInput(name, list(arr.shape), dtype)
            if self.scheme == "http":
                inp.set_data_from_numpy(arr, binary_data=True)
            else:
                inp.set_data_from_numpy(arr)
            placeholders.append(inp)
        return placeholders

    def __call__(
        self, *args: Any
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Run synchronous inference.

        Args:
            *args: Arrays corresponding to model inputs.

        Returns:
            A tuple of (bboxes, confidences, class_indices).

        Raises:
            RuntimeError: If required outputs are missing.
        """
        inputs = self._pack_inputs(*args)
        rsp = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=self._requested_outputs_objs,
        )
        boxes = rsp.as_numpy("boxes")
        num_dets = (
            rsp.as_numpy("num_dets")
            if "num_dets" in self.output_names
            else None
        )
        if boxes is None:
            raise RuntimeError("Missing 'boxes'")

        boxes2d = boxes[0] if boxes.ndim == 3 else boxes
        n = (
            int(np.count_nonzero(boxes2d[:, 4] > 0))
            if num_dets is None
            else int(np.asarray(num_dets).reshape(-1)[0])
        )
        n = max(0, min(n, boxes2d.shape[0]))
        valid = boxes2d[:n].astype(np.float32, copy=False)
        return (
            valid[:, :4].tolist(),
            valid[:, 4].astype(float).tolist(),
            valid[:, 5].astype(int).tolist(),
        )

    def infer_async(
        self, *args: Any, timeout_ms: int = 0
    ) -> Optional[Tuple[List[List[float]], List[float], List[int]]]:
        """
        Run asynchronous inference for gRPC or fallback to sync for HTTP.

        Args:
            *args: Arrays corresponding to model inputs.
            timeout_ms: Maximum time to wait for completion. If 0, wait indefinitely.

        Returns:
            Tuple of (bboxes, confidences, class_indices) or None if timed out (gRPC only).

        Raises:
            RuntimeError: If the underlying client reports an error.
        """
        if self.scheme != "grpc":
            return self.__call__(*args)
        from threading import Event

        done = Event()
        result: Dict[str, Any] = {"ok": False, "rsp": None, "err": None}

        def cb(
            user_data: Any, result_: Any, error_: Optional[BaseException]
        ) -> None:
            if error_ is not None:
                result["err"] = error_
            else:
                result["ok"] = True
                result["rsp"] = result_
            done.set()

        inputs = self._pack_inputs(*args)
        self.client.async_infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=self._requested_outputs_objs,
            callback=cb,
        )
        if timeout_ms > 0:
            finished = done.wait(timeout_ms / 1000.0)
            if not finished:
                return None
        else:
            done.wait()
        if not result["ok"]:
            raise RuntimeError(result["err"])
        rsp = result["rsp"]
        boxes = rsp.as_numpy("boxes")
        num_dets = (
            rsp.as_numpy("num_dets")
            if "num_dets" in self.output_names
            else None
        )
        if boxes is None:
            return ([], [], [])
        boxes2d = boxes[0] if boxes.ndim == 3 else boxes
        n = (
            int(np.count_nonzero(boxes2d[:, 4] > 0))
            if num_dets is None
            else int(np.asarray(num_dets).reshape(-1)[0])
        )
        n = max(0, min(n, boxes2d.shape[0]))
        valid = boxes2d[:n].astype(np.float32, copy=False)
        return (
            valid[:, :4].tolist(),
            valid[:, 4].astype(float).tolist(),
            valid[:, 5].astype(int).tolist(),
        )

    def _get_labels(self) -> Optional[List[str]]:
        """
        Load class labels from the model directory if configured.

        The Triton ensemble model output may specify `label_filename` for the first
        output. This method attempts to read that file from either the Docker
        path or the local project path.

        Returns:
            A list of labels if found, otherwise None.
        """
        label_filename = self.config["output"][0]["label_filename"]
        docker_file_path = (
            f"/root/app/triton/{self.model_name}/{label_filename}"
        )
        local_file_path = os.path.join(
            os.path.abspath(os.getcwd()),
            f"triton/{self.model_name}/{label_filename}",
        )

        if os.path.isfile(docker_file_path):
            with open(docker_file_path, "r") as file:
                labels = file.read().splitlines()
        elif os.path.isfile(local_file_path):
            with open(local_file_path, "r") as file:
                labels = file.read().splitlines()
        else:
            labels = None

        return labels


class Annotator:
    """
    Annotate frames with boxes/labels or overlay a Santa hat.
    """

    def __init__(
        self,
        labels: Optional[List[str]],
        width: int,
        height: int,
        classes: Optional[Sequence[int]] = None,
        santa_hat_plugin: bool = False,
        santa_hat_path: str = "./images/santa_hat.png",
        santa_hat_mask_path: str = "./images/santa_hat_mask.png",
        hat_scale: float = 0.7,
        hat_y_offset: int = 20,
        hat_x_offset: int = 0,
    ) -> None:
        """
        Initialize the annotator.

        Args:
            labels: Optional list of class labels.
            width: Frame width in pixels.
            height: Frame height in pixels.
            classes: Optional whitelist of class indices to draw.
            santa_hat_plugin: Whether to draw a Santa hat on the best detection.
            santa_hat_path: Path to the hat image.
            santa_hat_mask_path: Path to the hat mask image.
            hat_scale: Relative hat width as a fraction of the bbox width.
            hat_y_offset: Vertical offset from the top of the bbox.
            hat_x_offset: Horizontal offset applied to the hat placement.
        """
        self.width, self.height = width, height
        self.labels = labels
        self._allow_all = (classes is None) or (len(classes) == 0)
        self._allow = set(classes) if not self._allow_all else None

        n_colors = len(labels) if labels else 100
        rng = np.random.default_rng(123)
        self.colors: List[List[int]] = (
            (rng.random((n_colors, 3)) * 255).astype(np.uint8).tolist()
        )

        self.santa_hat_plugin = santa_hat_plugin
        self.santa_hat = cv2.imread(santa_hat_path)
        self.santa_hat_mask = cv2.imread(santa_hat_mask_path)
        self.hat_scale = float(hat_scale)
        self.hat_y_offset = int(hat_y_offset)
        self.hat_x_offset = int(hat_x_offset)

    def __call__(
        self,
        frame: np.ndarray,
        bboxes: List[List[float]],
        confs: List[float],
        indexes: List[int],
    ) -> np.ndarray:
        """
        Annotate a frame.

        Args:
            frame: Frame to draw on (BGR).
            bboxes: Bounding boxes [[x1,y1,x2,y2], ...] normalized or absolute.
            confs: Confidence scores.
            indexes: Class indices.

        Returns:
            The annotated frame.
        """
        if not bboxes:
            return frame

        if not self._allow_all:
            keep = [
                i for i, cls in enumerate(indexes) if int(cls) in self._allow
            ]
        else:
            keep = list(range(len(bboxes)))

        if not keep:
            return frame

        if not self.santa_hat_plugin:
            for i in keep:
                bb = self._to_pixels(bboxes[i])
                xmin, ymin, xmax, ymax = [int(v) for v in bb]
                cls_id = int(indexes[i])
                color = tuple(
                    int(c) for c in self.colors[cls_id % len(self.colors)]
                )
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                lbl = (
                    self.labels[cls_id]
                    if (self.labels and cls_id < len(self.labels))
                    else str(cls_id)
                )
                cv2.putText(
                    frame,
                    f"{lbl} ({confs[i]:.2f})",
                    (xmin, max(0, ymin - 5)),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.9,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            return frame

        best_idx = max(keep, key=lambda i: confs[i])
        bb = self._to_pixels(bboxes[best_idx])
        return self._overlay_hat(frame, bb)

    def _to_pixels(self, bb: List[float]) -> List[int]:
        """
        Convert a bbox to absolute pixel coordinates within frame bounds.

        Args:
            bb: Bounding box [x1, y1, x2, y2], normalized or absolute.

        Returns:
            Bounding box as [x1, y1, x2, y2] integers in pixel space.
        """
        x1, y1, x2, y2 = bb
        if max(x1, y1, x2, y2) <= 1.5:
            x1 *= self.width
            x2 *= self.width
            y1 *= self.height
            y2 *= self.height

        x1 = int(max(0, min(self.width - 1, round(x1))))
        y1 = int(max(0, min(self.height - 1, round(y1))))
        x2 = int(max(0, min(self.width - 1, round(x2))))
        y2 = int(max(0, min(self.height - 1, round(y2))))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    def _overlay_hat(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Overlay the Santa hat image on a frame given a bounding box.

        Args:
            frame: Target frame (BGR).
            bbox: Bounding box [x1, y1, x2, y2] in pixels.

        Returns:
            Frame with hat composited if possible.
        """
        if self.santa_hat is None or self.santa_hat_mask is None:
            return frame

        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)

        resize_width = max(1, int(bw * self.hat_scale))

        x = x1 + (bw - resize_width) // 2 + self.hat_x_offset
        y = y1 + self.hat_y_offset

        santa_hat = imutils.resize(self.santa_hat.copy(), width=resize_width)
        santa_hat_mask = imutils.resize(
            self.santa_hat_mask.copy(), width=resize_width
        )
        hat_h, hat_w = santa_hat.shape[:2]

        mask_bool = santa_hat_mask[:, :, 0] == 0
        mask_rgb = np.stack([mask_bool, mask_bool, mask_bool], axis=2)

        def clamp(v: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, v))

        x0 = clamp(x, 0, self.width)
        y0 = clamp(y - hat_h, 0, self.height)
        x1p = clamp(x + hat_w, 0, self.width)
        y1p = clamp(y, 0, self.height)

        dst_w = max(0, x1p - x0)
        dst_h = max(0, y1p - y0)
        if dst_w == 0 or dst_h == 0:
            return frame

        sx0 = x0 - x
        sy0 = y0 - (y - hat_h)
        sx1 = sx0 + dst_w
        sy1 = sy0 + dst_h

        hat_roi = santa_hat[sy0:sy1, sx0:sx1, :]
        mask_roi = mask_rgb[sy0:sy1, sx0:sx1, :]
        inv_mask_roi = ~mask_roi

        dst = frame[y0:y1p, x0:x1p, :]
        np.multiply(dst, inv_mask_roi, out=dst, casting="unsafe")
        dst += (hat_roi * mask_roi).astype(dst.dtype, copy=False)
        frame[y0:y1p, x0:x1p, :] = dst
        return frame


def build_ffmpeg_command(
    rtmp_url: str,
    w: int,
    h: int,
    fps: int,
    audio_enabled: bool,
    audio_device: str,
) -> List[str]:
    """
    Build an ffmpeg command for RTMP streaming.

    Args:
        rtmp_url: RTMP destination URL.
        w: Frame width in pixels.
        h: Frame height in pixels.
        fps: Frames per second.
        audio_enabled: Whether to include audio input.
        audio_device: ALSA device string if audio is enabled.

    Returns:
        The ffmpeg command as a list of arguments.
    """
    common_in = [
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-rtbufsize",
        "100k",
        "-thread_queue_size",
        "1024",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-use_wallclock_as_timestamps",
        "1",
        "-i",
        "-",
    ]

    video_encode = [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(max(1, int(fps * 2))),
        "-keyint_min",
        str(max(1, int(fps * 2))),
        "-sc_threshold",
        "0",
        "-x264-params",
        "vbv-bufsize=1000:vbv-maxrate=2500",
        "-maxrate",
        "2500k",
        "-bufsize",
        "1000k",
        "-fps_mode",
        "passthrough",
        "-fflags",
        "nobuffer",
        "-flush_packets",
        "1",
        "-rtmp_live",
        "live",
        "-f",
        "flv",
        rtmp_url,
    ]

    if not audio_enabled or not audio_device:
        return (
            ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
            + common_in
            + ["-an"]
            + video_encode
        )

    audio_in = ["-thread_queue_size", "1024", "-f", "alsa", "-i", audio_device]

    audio_enc = ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]

    return (
        ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
        + common_in
        + audio_in
        + audio_enc
        + video_encode
    )


def capture_thread(
    stop_evt: threading.Event,
    cap: cv2.VideoCapture,
    latest: LatestFrame,
    target_fps: int,
) -> None:
    """
    Continuously capture frames and publish the latest one.

    Args:
        stop_evt: Event to signal thread termination.
        cap: OpenCV capture device.
        latest: Shared latest-frame container.
        target_fps: Desired capture rate.
    """
    period = 1.0 / max(1, target_fps)
    next_t = time.time()
    while not stop_evt.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        latest.set(frame)
        now = time.time()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += period


def inference_thread(
    stop_evt: threading.Event,
    latest: LatestFrame,
    dets: AtomicDetections,
    model: TritonClient,
    input_w: int,
    input_h: int,
    target_period_frames: int = 2,
    adaptive_latency_ms: int = 80,
    use_async: bool = False,
) -> None:
    """
    Run periodic inference and update detections.

    Args:
        stop_evt: Event to signal termination.
        latest: Source of the most recent frame.
        dets: Destination for detection results.
        model: Triton client.
        input_w: Model input width.
        input_h: Model input height.
        target_period_frames: Frame period between inferences at 30 FPS baseline.
        adaptive_latency_ms: Target latency for adaptive pacing.
        use_async: Whether to use asynchronous inference.
    """
    stride = max(1, target_period_frames)
    last_infer_t = 0.0
    while not stop_evt.is_set():
        frame, ts = latest.get()
        if frame is None:
            time.sleep(0.005)
            continue
        if frame.shape[1] != input_w or frame.shape[0] != input_h:
            frame = cv2.resize(
                frame, (input_w, input_h), interpolation=cv2.INTER_AREA
            )
        now = time.time()
        if (now - last_infer_t) < (stride / 30.0):
            time.sleep(0.001)
            continue
        last_infer_t = now
        t0 = time.time()
        try:
            if use_async:
                out = model.infer_async(
                    frame[None, :, :, :], timeout_ms=adaptive_latency_ms
                )
                if out is None:
                    stride = min(6, stride + 1)
                    continue
                bboxes, confs, indexes = out
            else:
                bboxes, confs, indexes = model(frame[None, :, :, :])
            dets.set(bboxes, confs, indexes)
        except Exception:
            pass
        elapsed_ms = (time.time() - t0) * 1000.0
        if elapsed_ms > adaptive_latency_ms and stride < 6:
            stride += 1
        elif elapsed_ms < adaptive_latency_ms * 0.5 and stride > 1:
            stride -= 1


def encoder_thread(
    stop_evt: threading.Event,
    w: int,
    h: int,
    fps: int,
    p: subprocess.Popen,
    latest: LatestFrame,
    dets: AtomicDetections,
    annotator: Annotator,
) -> None:
    """
    Consume frames, annotate, and pipe to ffmpeg stdin.

    Args:
        stop_evt: Event to signal termination.
        w: Frame width in pixels.
        h: Frame height in pixels.
        fps: Output frame rate.
        p: Subprocess with stdin for raw video.
        latest: Source of the most recent frame.
        dets: Source of detections.
        annotator: Callable annotator that returns an annotated frame.
    """
    frame_bytes = w * h * 3
    fifo: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=3)

    def producer() -> None:
        """
        Periodically push latest frames into a small FIFO.
        """
        period = 1.0 / max(1, fps)
        next_t = time.time()
        while not stop_evt.is_set():
            frame, _ = latest.get()
            if frame is None:
                time.sleep(0.002)
                continue
            try:
                fifo.put(frame, timeout=0.02)
            except queue.Full:
                try:
                    fifo.get_nowait()
                    fifo.put_nowait(frame)
                except Exception:
                    pass
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period

    prod_t = threading.Thread(target=producer, daemon=True)
    prod_t.start()

    while not stop_evt.is_set():
        try:
            frame = fifo.get(timeout=0.1)
        except queue.Empty:
            continue
        bboxes, confs, indexes, _ = dets.get()
        if bboxes:
            frame = annotator(frame, bboxes, confs, indexes)
        try:
            buf = frame.tobytes()
            if len(buf) != frame_bytes:
                continue
            if p.stdin is not None:
                p.stdin.write(buf)
        except Exception:
            break


def main():
    """
    Entrypoint to start capture, inference, and streaming pipeline.
    """
    parser = EnvArgumentParser()
    parser.add_arg("TRITON_URL", default="grpc://localhost:8001", d_type=str)
    parser.add_arg("MODEL_NAME", default="yolo11n", d_type=str)
    parser.add_arg("CLASSES", default=[], d_type=list)
    parser.add_arg("STREAM_IP", default="127.0.0.1", d_type=str)
    parser.add_arg("STREAM_PORT", default=1935, d_type=int)
    parser.add_arg("STREAM_APPLICATION", default="live", d_type=str)
    parser.add_arg("STREAM_KEY", default="stream", d_type=str)
    parser.add_arg("CAMERA_INDEX", default=0, d_type=int)
    parser.add_arg("CAMERA_WIDTH", default=640, d_type=int)
    parser.add_arg("CAMERA_HEIGHT", default=480, d_type=int)
    parser.add_arg("CAMERA_FPS", default=30, d_type=int)
    parser.add_arg("SANTA_HAT_PLUGIN", default=False, d_type=bool)
    parser.add_arg("AUDIO_ENABLED", default=False, d_type=bool)
    parser.add_arg("AUDIO_DEVICE", default="", d_type=str)
    parser.add_arg("SANTA_HAT_PLUGIN", default=False, d_type=bool)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, args.CAMERA_FPS)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    rtmp_url = f"rtmp://{args.STREAM_IP}:{args.STREAM_PORT}/{args.STREAM_APPLICATION}/{args.STREAM_KEY}"
    ffmpeg_cmd = build_ffmpeg_command(
        rtmp_url,
        args.CAMERA_WIDTH,
        args.CAMERA_HEIGHT,
        args.CAMERA_FPS,
        args.AUDIO_ENABLED,
        args.AUDIO_DEVICE,
    )
    p = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    model = TritonClient(
        url=args.TRITON_URL,
        model=args.MODEL_NAME,
        requested_outputs=["boxes", "num_dets"],
    )
    annotator = Annotator(
        labels=model.labels,
        width=args.CAMERA_WIDTH,
        height=args.CAMERA_HEIGHT,
        classes=args.CLASSES,
        santa_hat_plugin=args.SANTA_HAT_PLUGIN,
    )

    latest = LatestFrame()
    dets = AtomicDetections()
    stop_evt = threading.Event()

    def handle_sig(*_):
        stop_evt.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, handle_sig)

    t_cap = threading.Thread(
        target=capture_thread,
        args=(stop_evt, cap, latest, args.CAMERA_FPS),
        daemon=True,
    )
    t_inf = threading.Thread(
        target=inference_thread,
        args=(
            stop_evt,
            latest,
            dets,
            model,
            args.CAMERA_WIDTH,
            args.CAMERA_HEIGHT,
            2,
            80,
            False,
        ),
        daemon=True,
    )
    t_enc = threading.Thread(
        target=encoder_thread,
        args=(
            stop_evt,
            args.CAMERA_WIDTH,
            args.CAMERA_HEIGHT,
            args.CAMERA_FPS,
            p,
            latest,
            dets,
            annotator,
        ),
        daemon=True,
    )

    t_cap.start()
    t_inf.start()
    t_enc.start()

    try:
        while not stop_evt.is_set():
            if p.poll() is not None:
                stop_evt.set()
            time.sleep(0.2)
    finally:
        stop_evt.set()
        cap.release()
        try:
            if p.stdin:
                p.stdin.close()
        except Exception:
            pass
        try:
            p.terminate()
            p.wait(timeout=2)
        except Exception:
            p.kill()


if __name__ == "__main__":
    main()
