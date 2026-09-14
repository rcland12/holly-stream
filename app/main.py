import os
import subprocess
from ast import literal_eval
from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse

import cv2
import imutils
import numpy as np
from dotenv import load_dotenv
from libcamera import Transform
from picamera2 import Picamera2


class EnvArgumentParser:
    """
    A class for parsing environment variables as arguments with most Python types.
    """

    def __init__(self):
        self.dict: Dict[str, Any] = {}

    class _define_dict(dict):
        """
        A custom dictionary subclass for accessing arguments as attributes.
        """

        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def add_arg(
        self, variable: str, default: Any = None, d_type: Type = str
    ) -> None:
        """
        Add an argument to be parsed from an environment variable.

        Args:
            variable (str): The name of the environment variable.
            default (Any): The default value if the environment variable is not set.
            d_type (Type): The expected data type of the argument. Defaults to str.
        """
        env = os.environ.get(variable)
        if env is None:
            try:
                if isinstance(default, d_type):
                    value = default
                else:
                    raise TypeError(
                        f"The default value for {variable} cannot be cast to the data type provided."
                    )
            except TypeError:
                raise TypeError(
                    f"The type you provided for {variable} is not valid."
                )
        else:
            if callable(d_type):
                value = self._cast_type(env, d_type)
        self.dict[variable] = value

    @staticmethod
    def _cast_type(arg: str, d_type: Type) -> Any:
        """
        Cast the argument to the specified data type.

        Args:
            arg (str): The argument value as a string.
            d_type (Type): The desired data type.

        Returns:
            Any: The argument value casted to the specified data type.

        Raises:
            ValueError: If the argument does not match the given data type or is not supported.
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
                    f"Check the types entered for arugment {arg}:", str(e)
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
                    f"Check the types entered for arugment {arg}:", str(e)
                )

        return cast_value

    def parse_args(self) -> "_define_dict":
        """
        Parse the added arguments from the environment variables.

        Returns:
            _define_dict: A custom dictionary containing the parsed arguments.
        """
        return self._define_dict(self.dict)


class TritonClient:
    """
    A client class for interacting with Triton Inference Server.

    Args:
        url (str): The URL of the Triton Inference Server.
        model (str): The name of the model to be used for inference.

    Attributes:
        client (InferenceServerClient): The Triton Inference Server client instance.
        model_name (str): The name of the model being used.
        metadata (dict): The metadata of the model.
        config (dict): The configuration of the model.
        model_dims (Tuple[int, int]): The dimensions of the model input.
        classes (Optional[List[str]]): The list of class labels, if available.

    Raises:
        RuntimeError: If an unsupported protocol is used (other than HTTP or GRPC).
    """

    def __init__(self, url: str, model: str):
        """
        Initialize the TritonClient instance.

        Args:
            url (str): The URL of the Triton Inference Server.
            model (str): The name of the model to be used for inference.
        """
        parsed_url = urlparse(url)
        self.model_name: str = model

        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient, InferInput

            self.client: InferenceServerClient = InferenceServerClient(
                parsed_url.netloc
            )
            self.metadata: dict = self.client.get_model_metadata(
                self.model_name, as_json=True
            )

            def get_model_config(name: str) -> dict:
                return self.client.get_model_config(name, as_json=True)["config"]

        elif parsed_url.scheme == "http":
            from tritonclient.http import InferenceServerClient, InferInput

            self.client: InferenceServerClient = InferenceServerClient(
                parsed_url.netloc
            )
            self.metadata: dict = self.client.get_model_metadata(
                self.model_name
            )

            def get_model_config(name: str) -> dict:
                return self.client.get_model_config(name)

        else:
            raise RuntimeError("Unsupported protocol. Use HTTP or GRPC.")

        self._infer_input_cls = InferInput
        self._get_model_config_fn = get_model_config
        self.config: dict = get_model_config(self.model_name)
        self.model_dims: Tuple[int, int] = self._get_dims()
        self.classes: Optional[List[str]] = self._get_classes()

    def __call__(
        self, frame: np.ndarray
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Run inference on an RGB frame and parse the NMS detections.

        Expects the model to produce a single [1, N, 6] output where each row is
        [x1, y1, x2, y2, score, class] in model input pixels, zero-padded to N rows.

        Args:
            frame (np.ndarray): RGB frame with shape [H, W, 3] and dtype uint8.

        Returns:
            Tuple containing:
                - List[List[float]]: Bounding boxes as [x1, y1, x2, y2] in frame pixels
                - List[float]: Confidence scores rounded to 2 decimal places
                - List[int]: Class indexes
        """
        inputs = self._create_inputs(np.ascontiguousarray(frame))
        response = self.client.infer(model_name=self.model_name, inputs=inputs)

        detections = response.as_numpy(self.metadata["outputs"][0]["name"])[0]
        detections = detections[detections[:, 4] > 0]

        boxes = self._rescale_boxes(detections[:, :4], frame.shape[:2])
        bboxes = boxes.tolist()
        confs = [round(float(score), 2) for score in detections[:, 4]]
        indexes = detections[:, 5].astype(int).tolist()

        return bboxes, confs, indexes

    def _rescale_boxes(
        self, boxes: np.ndarray, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Rescale boxes from the letterboxed model input back to the original frame.

        Args:
            boxes (np.ndarray): Array of shape [N, 4] as [x1, y1, x2, y2] in model input pixels.
            original_shape (Tuple[int, int]): (height, width) of the original frame.

        Returns:
            np.ndarray: Rescaled boxes clipped to the frame, shape [N, 4].
        """
        orig_h, orig_w = original_shape
        model_h, model_w = self.model_dims

        # Mirrors the letterbox in triton/repository/preprocess/1/model.py
        scale = min(model_h / orig_h, model_w / orig_w)
        pad_top = (model_h - int(round(orig_h * scale))) // 2
        pad_left = (model_w - int(round(orig_w * scale))) // 2

        rescaled = boxes.astype(np.float32)
        rescaled[:, [0, 2]] = ((rescaled[:, [0, 2]] - pad_left) / scale).clip(0, orig_w)
        rescaled[:, [1, 3]] = ((rescaled[:, [1, 3]] - pad_top) / scale).clip(0, orig_h)

        return rescaled

    def _create_inputs(self, *args):
        """
        Create input placeholders for the model.

        Args:
            *args: The input arguments for the model.

        Returns:
            List[InferInput]: The list of input placeholders.

        Raises:
            RuntimeError: If no inputs are provided or if the number of inputs does not match the expected number.
        """
        args_len = len(args)
        if not args_len:
            raise RuntimeError("No inputs provided.")

        input_specs = self.metadata["inputs"]
        if args_len != len(input_specs):
            raise RuntimeError(
                f"Expected {len(input_specs)} inputs, got {args_len}."
            )

        # Use the actual array shape, the ensemble input has variable height/width
        placeholders = []
        for spec, value in zip(input_specs, args):
            placeholder = self._infer_input_cls(
                spec["name"], list(value.shape), spec["datatype"]
            )
            placeholder.set_data_from_numpy(value)
            placeholders.append(placeholder)

        return placeholders

    def _get_classes(self) -> Optional[List[str]]:
        """
        Get the class labels of the model, if available.

        Returns:
            Optional[List[str]]: The list of class labels, or None if not available.
        """
        label_filename = next(
            (
                o["label_filename"]
                for o in self.config["output"]
                if o.get("label_filename")
            ),
            None,
        )
        if label_filename is None:
            return None

        # The model repository is mounted read-only at /root/app/triton/repository in compose.yml
        candidate_paths = [
            f"/root/app/triton/repository/{self.model_name}/{label_filename}",
            os.path.join(
                os.getcwd(),
                f"triton/repository/{self.model_name}/{label_filename}",
            ),
        ]

        for path in candidate_paths:
            if os.path.isfile(path):
                with open(path, "r") as file:
                    return [
                        line for line in file.read().splitlines() if line.strip()
                    ]

        raise FileNotFoundError(
            f"Could not find label file '{label_filename}' for model '{self.model_name}'. Searched: {candidate_paths}"
        )

    def _get_dims(self) -> Tuple[int, int]:
        """
        Get the (height, width) the frame is letterboxed to, read from the first
        ensemble step's output (the preprocess model).

        Returns:
            Tuple[int, int]: The dimensions of the model input.
        """
        try:
            first_step = self.config["ensemble_scheduling"]["step"][0]["model_name"]
            config = self._get_model_config_fn(first_step)
            return tuple(int(d) for d in config["output"][0]["dims"][2:4])
        except Exception:
            return (640, 640)


class Annotator:
    """
    A class for annotating frames with bounding boxes, class labels, and confidence scores.

    Args:
        classes (List[str]): A list of class labels.
        width (int): The width of the frame. Defaults to 1280.
        height (int): The height of the frame. Defaults to 720.
        santa_hat_plugin (bool): Indicates whether to use the Santa hat plugin. Defaults to False.

    Attributes:
        width (int): The width of the frame.
        height (int): The height of the frame.
        classes (List[str]): A list of class labels.
        colors (List[Tuple[float, float, float]]): A list of randomly generated colors for each class.
        santa_hat (np.ndarray): The Santa hat image.
        santa_hat_mask (np.ndarray): The Santa hat mask image.
        santa_hat_plugin (bool): Indicates whether to use the Santa hat plugin.
    """

    def __init__(
        self,
        classes: List[str],
        width: int = 1280,
        height: int = 720,
        santa_hat_plugin: bool = False,
    ):
        """
        Initialize the Annotator instance.

        Args:
            classes (List[str]): A list of class labels.
            width (int): The width of the frame. Defaults to 1280.
            height (int): The height of the frame. Defaults to 720.
            santa_hat_plugin (bool): Indicates whether to use the Santa hat plugin. Defaults to False.
        """
        self.width = width
        self.height = height
        self.classes = classes
        self.colors = list(np.random.rand(len(self.classes), 3) * 255)
        self.santa_hat = cv2.imread("images/santa_hat.png")
        self.santa_hat_mask = cv2.imread("images/santa_hat_mask.png")
        self.santa_hat_plugin = santa_hat_plugin

    def __call__(
        self,
        frame: np.ndarray,
        bboxes: List[List[float]],
        confs: List[float],
        indexes: List[int],
    ) -> np.ndarray:
        """
        Annotate the frame with bounding boxes, class labels, and confidence scores.

        Args:
            frame (np.ndarray): The input frame.
            bboxes (List[List[float]]): A list of bounding box coordinates.
            confs (List[float]): A list of confidence scores.
            indexes (List[int]): A list of class indexes.

        Returns:
            np.ndarray: The annotated frame.
        """
        if not self.santa_hat_plugin:
            for i in range(len(bboxes)):
                xmin, ymin, xmax, ymax = [int(j) for j in bboxes[i]]
                color = self.colors[indexes[i]]
                frame = cv2.rectangle(
                    img=frame,
                    pt1=(xmin, ymin),
                    pt2=(xmax, ymax),
                    color=color,
                    thickness=2,
                )

                frame = cv2.putText(
                    img=frame,
                    text=f"{self.classes[indexes[i]]} ({str(confs[i])})",
                    org=(xmin, ymin - 5),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=0.75,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            return frame

        else:
            max_index = max(range(len(confs)), key=confs.__getitem__)
            return self._overlay_obj(frame, bboxes[max_index].copy())

    def _overlay_obj(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Overlay the Santa hat on the detected object.

        Args:
            frame (np.ndarray): The input frame.
            bbox (List[float]): The bounding box coordinates of the detected object.

        Returns:
            np.ndarray: The frame with the Santa hat overlaid on the detected object.
        """
        bbox = [int(i) for i in bbox]
        x, y = bbox[0], bbox[1] + 20

        resize_width = bbox[2] - bbox[0]
        santa_hat = imutils.resize(self.santa_hat.copy(), width=resize_width)
        santa_hat_mask = imutils.resize(
            self.santa_hat_mask.copy(), width=resize_width
        )
        hat_height, hat_width = santa_hat.shape[0], santa_hat.shape[1]

        mask_boolean = santa_hat_mask[:, :, 0] == 0
        mask_rgb_boolean = np.stack(
            [mask_boolean, mask_boolean, mask_boolean], axis=2
        )

        if x >= 0 and y >= 0:
            h = hat_height - max(0, y + hat_height - self.height)
            w = hat_width - max(0, x + hat_width - self.width)
            frame[y - h : y, x : x + w, :] = (
                frame[y - h : y, x : x + w, :] * ~mask_rgb_boolean[0:h, 0:w, :]
                + (santa_hat * mask_rgb_boolean)[0:h, 0:w, :]
            )

        elif x < 0 and y < 0:
            h = hat_height + y
            w = hat_width + x
            frame[0 : 0 + h, 0 : 0 + w, :] = (
                frame[0 : 0 + h, 0 : 0 + w, :]
                * ~mask_rgb_boolean[
                    hat_height - h : hat_height, hat_width - w : hat_width, :
                ]
                + (santa_hat * mask_rgb_boolean)[
                    hat_height - h : hat_height, hat_width - w : hat_width, :
                ]
            )

        elif x < 0 and y >= 0:
            h = hat_height - max(0, y + hat_height - self.height)
            w = hat_width + x
            frame[y : y + h, 0 : 0 + w, :] = (
                frame[y : y + h, 0 : 0 + w, :]
                * ~mask_rgb_boolean[0:h, hat_width - w : hat_width, :]
                + (santa_hat * mask_rgb_boolean)[
                    0:h, hat_width - w : hat_width, :
                ]
            )

        elif x >= 0 and y < 0:
            h = hat_height + y
            w = hat_width - max(0, x + hat_width - self.width)
            frame[0 : 0 + h, x : x + w, :] = (
                frame[0 : 0 + h, x : x + w, :]
                * ~mask_rgb_boolean[hat_height - h : hat_height, 0:w, :]
                + (santa_hat * mask_rgb_boolean)[
                    hat_height - h : hat_height, 0:w, :
                ]
            )

        return frame


def main(
    triton_url: str,
    model_name: str,
    stream_ip: str,
    stream_port: int,
    stream_application: str,
    stream_key: str,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
    camera_rotation: int,
    camera_hflip: bool,
    camera_vflip: bool,
    santa_hat_plugin: bool,
    confidence_threshold: float,
    classes: List[int],
) -> None:
    """
    Main function to run the RTMP stream, object detection and annotation pipeline.

    Args:
        triton_url (str): The URL of the Triton server.
        model_name (str): The name of the model to use for object detection.
        stream_ip (str): The IP address of the RTMP stream server.
        stream_port (int): The port number of the RTMP stream server.
        stream_application (str): The application name for the RTMP stream.
        stream_key (str): The stream key for the RTMP stream.
        camera_width (int): The width of the camera frame.
        camera_height (int): The height of the camera frame.
        camera_fps (int): The frames-per-second to use on camera.
        camera_rotation (int): Image rotation, 0 or 180 (180 == hflip + vflip).
        camera_hflip (bool): Toggles a horizontal flip on top of the rotation.
        camera_vflip (bool): Toggles a vertical flip on top of the rotation.
        santa_hat_plugin (bool): Indicates whether to use the Santa hat plugin.
        confidence_threshold (float): Minimum score to draw. The model already drops scores below 0.25.
        classes (List[int]): Class indexes to keep when drawing detections. An empty list keeps all classes.

    Returns:
        None
    """
    rtmp_url = "rtmp://{}:{}/{}/{}".format(
        stream_ip, stream_port, stream_application, stream_key
    )

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "{}x{}".format(camera_width, camera_height),
        "-r",
        str(camera_fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-f",
        "flv",
        rtmp_url,
    ]

    model = TritonClient(triton_url, model_name)

    annotator = Annotator(
        model.classes, camera_width, camera_height, santa_hat_plugin
    )

    # Same orientation rules as stream.sh: a 180 rotation is hflip + vflip, and
    # CAMERA_HFLIP/CAMERA_VFLIP toggle those flips so every orientation is
    # reachable. 90/270 would need a real rotate the ISP cannot do.
    if camera_rotation not in (0, 180):
        print(
            f"[WARN] CAMERA_ROTATION={camera_rotation} is not supported (expected 0 or 180); treating as 0."
        )
        camera_rotation = 0
    hflip = (1 if camera_rotation == 180 else 0) ^ int(camera_hflip)
    vflip = (1 if camera_rotation == 180 else 0) ^ int(camera_vflip)
    print(f"[INFO] Orientation: hflip={hflip} vflip={vflip}")

    camera = Picamera2()
    camera.configure(
        camera.create_video_configuration(
            main={"size": (camera_width, camera_height), "format": "BGR888"},
            transform=Transform(hflip=hflip, vflip=vflip),
        )
    )
    camera.controls.Brightness = 0.2

    period = 10
    tracking_index = 0
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    try:
        camera.start()

        while True:
            frame = camera.capture_array()[:, :, :3]

            if tracking_index % period == 0:
                # The BGR888 format gives RGB-ordered pixels, which is what the model expects
                bboxes, confs, indexes = model(frame)
                keep = [
                    i
                    for i, (conf, index) in enumerate(zip(confs, indexes))
                    if conf >= confidence_threshold
                    and (not classes or index in classes)
                ]
                bboxes = [bboxes[i] for i in keep]
                confs = [confs[i] for i in keep]
                indexes = [indexes[i] for i in keep]
                tracking_index = 0

            if bboxes:
                frame = annotator(frame, bboxes, confs, indexes)
            tracking_index += 1
            process.stdin.write(frame.tobytes())

    finally:
        camera.stop()
        process.stdin.close()
        process.wait()

    return


if __name__ == "__main__":
    load_dotenv()
    parser = EnvArgumentParser()
    parser.add_arg("TRITON_URL", default="http://localhost:8000", d_type=str)
    parser.add_arg("MODEL_NAME", default="yolo11", d_type=str)
    parser.add_arg("STREAM_IP", default="127.0.0.1", d_type=str)
    parser.add_arg("STREAM_PORT", default=1935, d_type=int)
    parser.add_arg("STREAM_APPLICATION", default="live", d_type=str)
    parser.add_arg("STREAM_KEY", default="stream", d_type=str)
    parser.add_arg("CAMERA_WIDTH", default=640, d_type=int)
    parser.add_arg("CAMERA_HEIGHT", default=480, d_type=int)
    parser.add_arg("CAMERA_FPS", default=30, d_type=int)
    parser.add_arg("CAMERA_ROTATION", default=180, d_type=int)
    parser.add_arg("CAMERA_HFLIP", default=False, d_type=bool)
    parser.add_arg("CAMERA_VFLIP", default=False, d_type=bool)
    parser.add_arg("SANTA_HAT_PLUGIN", default=False, d_type=bool)
    parser.add_arg("CONFIDENCE_THRESHOLD", default=0.25, d_type=float)
    parser.add_arg("CLASSES", default=[], d_type=list)
    args = parser.parse_args()

    main(
        triton_url=args.TRITON_URL,
        model_name=args.MODEL_NAME,
        stream_ip=args.STREAM_IP,
        stream_port=args.STREAM_PORT,
        stream_application=args.STREAM_APPLICATION,
        stream_key=args.STREAM_KEY,
        camera_width=args.CAMERA_WIDTH,
        camera_height=args.CAMERA_HEIGHT,
        camera_fps=args.CAMERA_FPS,
        camera_rotation=args.CAMERA_ROTATION,
        camera_hflip=args.CAMERA_HFLIP,
        camera_vflip=args.CAMERA_VFLIP,
        santa_hat_plugin=args.SANTA_HAT_PLUGIN,
        confidence_threshold=args.CONFIDENCE_THRESHOLD,
        classes=args.CLASSES,
    )
