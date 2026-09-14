import argparse
import os
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np


class TritonClient:
    """
    Thin wrapper around Triton Inference Server HTTP/GRPC clients for the yolo11 ensemble.

    Attributes:
        model_name: Name of the Triton model.
        scheme: Connection scheme, either "http" or "grpc".
        client: Underlying Triton client instance.
        metadata: Model metadata as returned by Triton.
        config: Model config as returned by Triton.
        model_dims: (height, width) the frame is letterboxed to by the preprocess model.
        classes: Optional list of class labels loaded from the model repository.
    """

    def __init__(self, url: str, model: str) -> None:
        """
        Initialize the Triton client.

        Args:
            url: Server URL with scheme, e.g., "http://host:8000" or "grpc://host:8001".
            model: Model name deployed on Triton.
        """
        self.model_name: str = model

        parsed_url = urlparse(url)
        self.scheme = parsed_url.scheme.lower()

        if self.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)
            self.metadata = self.client.get_model_metadata(
                self.model_name, as_json=True
            )

        elif self.scheme == "http":
            from tritonclient.http import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)
            self.metadata = self.client.get_model_metadata(self.model_name)

        else:
            raise RuntimeError(
                "Unsupported protocol. Use http://host:port or grpc://host:port"
            )

        self._InferInput = InferInput
        self.config = self._get_model_config(self.model_name)
        self.model_dims: Tuple[int, int] = self._get_dims()
        self.classes: Optional[List[str]] = self._get_classes()

    def __call__(
        self, image: np.ndarray
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Run inference on an RGB image and parse the NMS detections.

        Expects a single [1, N, 6] output where each row is [x1, y1, x2, y2, score, class]
        in model input pixels, zero-padded to N rows.

        Args:
            image: RGB image with shape [H, W, 3] and dtype uint8.

        Returns:
            A tuple of (bboxes, confs, indexes):
                bboxes: List of [x1, y1, x2, y2] in original image pixels.
                confs: List of confidence scores.
                indexes: List of integer class indices.
        """
        input_spec = self.metadata["inputs"][0]
        image = np.ascontiguousarray(image, dtype=np.uint8)
        inp = self._InferInput(
            input_spec["name"], list(image.shape), input_spec["datatype"]
        )
        inp.set_data_from_numpy(image)

        response = self.client.infer(model_name=self.model_name, inputs=[inp])

        detections = response.as_numpy(self.metadata["outputs"][0]["name"])[0]
        detections = detections[detections[:, 4] > 0]
        print(detections[:, 4])

        boxes = self._rescale_boxes(detections[:, :4], image.shape[:2])

        return (
            boxes.tolist(),
            detections[:, 4].tolist(),
            detections[:, 5].astype(int).tolist(),
        )

    def _rescale_boxes(
        self, boxes: np.ndarray, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Rescale boxes from the letterboxed model input back to the original image.

        Mirrors the letterbox in repository/preprocess/1/model.py.

        Args:
            boxes: Array of shape [N, 4] as [x1, y1, x2, y2] in model input pixels.
            original_shape: (height, width) of the original image.

        Returns:
            Rescaled boxes clipped to the image, shape [N, 4].
        """
        orig_h, orig_w = original_shape
        model_h, model_w = self.model_dims

        scale = min(model_h / orig_h, model_w / orig_w)
        pad_top = (model_h - int(round(orig_h * scale))) // 2
        pad_left = (model_w - int(round(orig_w * scale))) // 2

        rescaled = boxes.astype(np.float32)
        rescaled[:, [0, 2]] = ((rescaled[:, [0, 2]] - pad_left) / scale).clip(
            0, orig_w
        )
        rescaled[:, [1, 3]] = ((rescaled[:, [1, 3]] - pad_top) / scale).clip(
            0, orig_h
        )

        return rescaled

    def _get_model_config(self, name: str) -> Any:
        """
        Fetch a model config as a dict for either protocol.

        Args:
            name: Model name deployed on Triton.

        Returns:
            The model config dictionary.
        """
        if self.scheme == "grpc":
            return self.client.get_model_config(name, as_json=True)["config"]
        return self.client.get_model_config(name)

    def _get_dims(self) -> Tuple[int, int]:
        """
        Read the letterbox size from the first ensemble step's output (the preprocess model).

        Returns:
            (height, width) of the model input.
        """
        try:
            first_step = self.config["ensemble_scheduling"]["step"][0][
                "model_name"
            ]
            dims = self._get_model_config(first_step)["output"][0]["dims"]
            return int(dims[2]), int(dims[3])
        except Exception:
            return 640, 640

    def _get_classes(self) -> Optional[List[str]]:
        """
        Load class labels from the local model repository next to this script.

        Returns:
            A list of class labels if found; otherwise, None.
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

        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"repository/{self.model_name}/{label_filename}",
        )
        if os.path.isfile(path):
            with open(path, "r") as f:
                return [line for line in f.read().splitlines() if line.strip()]
        return None


def draw_bounding_boxes(
    image: np.ndarray,
    bboxes: List[List[float]],
    confs: List[float],
    indexes: List[int],
    classes: Optional[List[str]] = None,
    output_path: str = "results.png",
) -> None:
    """
    Draw bounding boxes on an RGB image and write it to disk.

    Args:
        image: Input RGB image with shape [H, W, 3] and dtype uint8.
        bboxes: List of bounding boxes as [x1, y1, x2, y2].
        confs: Confidence scores corresponding to each bounding box.
        indexes: Class indices corresponding to each bounding box.
        classes: Optional list of class names indexed by class id.
        output_path: Output file path for the saved image.
    """
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

    for bbox, conf, class_idx in zip(bboxes, confs, indexes):
        x1, y1, x2, y2 = map(int, bbox)
        color = colors[class_idx % len(colors)]
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

        if classes and class_idx < len(classes):
            label = f"{classes[class_idx]}: {conf:.2f}"
        else:
            label = f"Class {class_idx}: {conf:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1,
        )
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    cv2.imwrite(output_path, img_with_boxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one image through the Triton yolo11 ensemble."
    )
    parser.add_argument("--url", default="grpc://rustypi6.home.arpa:8001")
    parser.add_argument("--model", default="yolo11")
    parser.add_argument(
        "--image", required=True, help="Path to an image of any size"
    )
    parser.add_argument("--output", default="results.png")
    args = parser.parse_args()

    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)

    client = TritonClient(url=args.url, model=args.model)
    bboxes, confs, indexes = client(image)

    draw_bounding_boxes(
        image, bboxes, confs, indexes, client.classes, args.output
    )
