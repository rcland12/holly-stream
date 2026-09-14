import json
from typing import Any, Dict, List

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Letterbox an RGB HWC uint8 frame of any size into a normalized 1x3xHxW FP32 tensor.

    The target size comes from this model's output dims in config.pbtxt, and the
    padding matches the rescaling done in app/main.py.
    """

    def initialize(self, args: Dict[str, Any]) -> None:
        model_config = json.loads(args["model_config"])
        self.input_name: str = model_config["input"][0]["name"]
        self.output_name: str = model_config["output"][0]["name"]
        self.model_height, self.model_width = [
            int(d) for d in model_config["output"][0]["dims"][2:4]
        ]
        self.pad_value = 114

    def _letterbox(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        r = min(self.model_height / height, self.model_width / width)
        new_height, new_width = int(round(height * r)), int(round(width * r))

        top = (self.model_height - new_height) // 2
        bottom = self.model_height - new_height - top
        left = (self.model_width - new_width) // 2
        right = self.model_width - new_width - left

        if (new_height, new_width) != (height, width):
            image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(self.pad_value, self.pad_value, self.pad_value),
        )

        # Triton serializes the numpy buffer directly, so the array must be contiguous.
        return np.ascontiguousarray(
            (image.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        )

    def execute(self, requests: List[Any]) -> List[Any]:
        responses = []
        for request in requests:
            image = pb_utils.get_input_tensor_by_name(
                request, self.input_name
            ).as_numpy()

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(self.output_name, self._letterbox(image))
                    ]
                )
            )

        return responses

    def finalize(self) -> None:
        print("Cleaning up preprocess model...", flush=True)
