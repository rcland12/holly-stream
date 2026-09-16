import os
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
from numpy.typing import NDArray


class TritonClient:
    """
    Thin wrapper around Triton Inference Server HTTP/GRPC clients.

    This client inspects the model's input and output metadata to prepare inputs
    and request specific outputs. It supports both HTTP and gRPC backends
    selected via the URL scheme.

    Attributes:
        model_name: Name of the Triton model.
        scheme: Connection scheme, either "http" or "grpc".
        client: Underlying Triton client instance.
        metadata: Model metadata as returned by Triton.
        config: Model config as returned by Triton.
        _InferInput: Triton input class bound to the selected backend.
        _InferRequestedOutput: Triton output request class bound to the backend.
        _input_schemas: Input specifications as (name, shape, dtype) tuples.
        _requested_outputs: Names of requested outputs.
        _requested_outputs_objs: Backend-specific output request objects.
        output_names: Alias for requested output names.
        classes: Optional list of class labels loaded from the model repository.
    """

    def __init__(
        self,
        url: str,
        model: str,
        requested_outputs: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize the Triton client.

        Args:
            url: Server URL with scheme, e.g., "http://host:8000" or "grpc://host:8001".
            model: Model name deployed on Triton.
            requested_outputs: Optional sequence of output tensor names to request.
        """
        self.model_name: str = model

        parsed_url = urlparse(url)
        scheme = parsed_url.scheme.lower()
        netloc = parsed_url.netloc

        if scheme not in ("grpc", "http"):
            raise RuntimeError(
                "Unsupported protocol. Use http://host:port or grpc://host:port"
            )

        self.scheme = scheme

        if scheme == "grpc":
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

        self._input_schemas = [
            (i["name"], [int(s) for s in i["shape"]], i["datatype"])
            for i in self.metadata["inputs"]
        ]

        if requested_outputs is None:
            self._requested_outputs = [
                o["name"] for o in self.metadata["outputs"]
            ]
        else:
            self._requested_outputs = list(requested_outputs)

        self._requested_outputs_objs: List[Any] = []
        for out_name in self._requested_outputs:
            if scheme == "http":
                self._requested_outputs_objs.append(
                    self._InferRequestedOutput(out_name, binary_data=True)
                )
            else:
                self._requested_outputs_objs.append(
                    self._InferRequestedOutput(out_name)
                )

        self.output_names = self._requested_outputs
        self.classes: Optional[List[str]] = self._get_classes()

    def __call__(
        self, *args: Any, original_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Run inference and parse NMS detection outputs.

        Expects the model to produce outputs:
        - "nms_num_dets": [1, 1] number of valid detections
        - "nms_boxes": [1, N, 4] bounding boxes as [x1, y1, x2, y2]
        - "nms_scores": [1, N] confidence scores
        - "nms_classes": [1, N] class indices

        Args:
            *args: Model inputs matching the Triton model's input schema.
            original_shape: Optional (height, width) of original image for rescaling boxes.

        Returns:
            A tuple of (bboxes, confs, indexes) for the first batch item:
                bboxes: List of [x1, y1, x2, y2].
                confs: List of confidence scores.
                indexes: List of integer class indices.

        Raises:
            RuntimeError: If expected outputs are missing.
        """
        inputs = self._pack_inputs(*args)

        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=self._requested_outputs_objs,
        )

        num_dets = response.as_numpy("nms_num_dets")
        boxes = response.as_numpy("nms_boxes")
        scores = response.as_numpy("nms_scores")
        classes = response.as_numpy("nms_classes")

        if boxes is None or scores is None or classes is None:
            raise RuntimeError("Triton response missing required NMS outputs.")

        n = (
            int(num_dets.reshape(-1)[0])
            if num_dets is not None
            else boxes.shape[1]
        )
        n = max(0, min(n, boxes.shape[1]))

        boxes_valid = boxes[0, :n].astype(np.float32, copy=False)
        scores_valid = scores[0, :n].astype(float, copy=False)
        print(scores_valid)
        classes_valid = classes[0, :n].astype(int, copy=False)

        if original_shape is not None:
            boxes_valid = self._rescale_boxes(boxes_valid, original_shape)

        bboxes = boxes_valid.tolist()
        confs = scores_valid.tolist()
        indexes = classes_valid.tolist()

        return bboxes, confs, indexes

    def _rescale_boxes(
        self, boxes: np.ndarray, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Rescale bounding boxes from model coordinates (640x640) to original image size.

        Accounts for letterbox/padding that maintains aspect ratio during preprocessing.

        Args:
            boxes: Array of shape [N, 4] with boxes as [x1, y1, x2, y2] in 640x640 space.
            original_shape: Tuple of (height, width) of the original image.

        Returns:
            Rescaled boxes array of shape [N, 4].
        """
        orig_h, orig_w = original_shape
        model_size = 640

        scale = min(model_size / orig_w, model_size / orig_h)

        pad_w = (model_size - orig_w * scale) / 2
        pad_h = (model_size - orig_h * scale) / 2

        rescaled = boxes.copy()
        rescaled[:, [0, 2]] = (rescaled[:, [0, 2]] - pad_w) / scale
        rescaled[:, [1, 3]] = (rescaled[:, [1, 3]] - pad_h) / scale

        return rescaled

    def infer_batch(self, *args: Any) -> Any:
        """
        Run inference and return the raw Triton response.

        This method does not parse outputs and is suitable for manual handling
        of batched outputs.

        Args:
            *args: Model inputs matching the Triton model's input schema.

        Returns:
            Backend-specific Triton inference response object.
        """
        inputs = self._pack_inputs(*args)

        return self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=self._requested_outputs_objs,
        )

    def _pack_inputs(self, *args: Any) -> List[Any]:
        """
        Convert Python and NumPy inputs into Triton input objects.

        Args:
            *args: Inputs corresponding to the model's input tensors, in order.

        Returns:
            A list of Triton input objects ready for inference.

        Raises:
            RuntimeError: If the number of inputs is incorrect or none provided.
            ValueError: If an input datatype cannot be mapped.
        """
        if not args:
            raise RuntimeError("No inputs provided.")

        if len(args) != len(self._input_schemas):
            raise RuntimeError(
                f"Expected {len(self._input_schemas)} inputs, got {len(args)}."
            )

        placeholders: List[Any] = []
        for (name, _, dtype), value in zip(self._input_schemas, args):
            arr = value if isinstance(value, np.ndarray) else np.asarray(value)
            np_dtype = self._triton_dtype_to_numpy(dtype)
            if arr.dtype != np_dtype:
                arr = arr.astype(np_dtype, copy=False)
            arr = np.ascontiguousarray(arr)
            inp = self._InferInput(name, list(arr.shape), dtype)
            if self.scheme == "http":
                inp.set_data_from_numpy(arr, binary_data=True)
            else:
                inp.set_data_from_numpy(arr)
            placeholders.append(inp)

        return placeholders

    @staticmethod
    def _triton_dtype_to_numpy(dtype: str) -> Any:
        """
        Map Triton datatype string to a NumPy dtype.

        Args:
            dtype: Triton datatype string such as "FP32" or "INT64".

        Returns:
            Corresponding NumPy dtype object.

        Raises:
            ValueError: If the datatype is not recognized.
        """
        import numpy as np

        mapping = {
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
        if dtype not in mapping:
            raise ValueError(f"Unhandled Triton dtype: {dtype}")
        return mapping[dtype]

    def _get_classes(self) -> Optional[List[str]]:
        """
        Load class labels from the model repository if available.

        The method inspects the model config to locate a label file and
        attempts to read it from common paths for both containerized and
        local development.

        Returns:
            A list of class labels if found; otherwise, None.
        """
        try:
            label_filename = self.config["output"][0]["label_filename"]
        except Exception:
            return None
        docker_file_path = (
            f"/root/app/triton/{self.model_name}/{label_filename}"
        )
        local_file_path = os.path.join(
            os.path.abspath(os.getcwd()),
            f"./{self.model_name}/{label_filename}",
        )
        path = (
            docker_file_path
            if os.path.isfile(docker_file_path)
            else local_file_path
        )
        if os.path.isfile(path):
            with open(path, "r") as f:
                return f.read().splitlines()
        return None


def draw_bounding_boxes(
    image: NDArray[np.uint8],
    bboxes: List[List[float]],
    confs: List[float],
    indexes: List[int],
    classes: Optional[List[str]] = None,
    output_path: str = "results.png",
) -> None:
    img_with_boxes = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    colors: List[Tuple[int, int, int]] = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (255, 165, 0),
        (0, 128, 255),
        (128, 255, 0),
    ]

    for _, (bbox, conf, class_idx) in enumerate(zip(bboxes, confs, indexes)):
        x1, y1, x2, y2 = map(int, bbox)
        color = colors[class_idx % len(colors)]
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

        if classes and 0 <= class_idx < len(classes):
            class_name = classes[class_idx]
            label = f"{class_name}: {conf:.2f}"
        else:
            label = f"Class {class_idx}: {conf:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        cv2.rectangle(
            img_with_boxes,
            (x1, max(0, y1 - text_height - baseline - 5)),
            (x1 + text_width, y1),
            color,
            -1,
        )
        cv2.putText(
            img_with_boxes,
            label,
            (
                x1,
                y1 - baseline - 2
                if y1 - baseline - 2 > 0
                else y1 + text_height + 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    cv2.imwrite(output_path, img_with_boxes)


if __name__ == "__main__":
    image_path = "../app/images/image_1280_720.png"
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    client = TritonClient(
        url="grpc://localhost:8001",
        model="yolo11",
    )

    bboxes, confs, indexes = client(image, original_shape=image.shape[:2])

    draw_bounding_boxes(
        image,
        bboxes,
        confs,
        indexes,
        client.classes,
    )
