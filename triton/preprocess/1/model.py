import json
import os
from ast import literal_eval
from typing import Any, Dict, List, Type, TypeVar, Union

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from c_python_backend_utils import InferenceRequest, InferenceResponse

T = TypeVar("T")


class EnvArgumentParser:
    """
    A parser for environment variables that supports type casting and default values.

    This class provides functionality similar to argparse.ArgumentParser but for
    environment variables, allowing type specification and default values.
    """

    def __init__(self):
        """
        Initialize an empty environment argument parser.
        """

        self.dict: Dict[str, Any] = {}

    class _define_dict(dict):
        """
        A custom dictionary class that allows attribute-style access to dictionary items.

        This enables accessing dictionary values using both square bracket notation
        and dot notation (e.g., dict['key'] or dict.key).
        """

        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def add_arg(
        self,
        variable: str,
        default: T = None,
        type: Union[Type[T], type] = str,
    ) -> None:
        """
        Add an environment variable argument with optional default value and type.

        Args:
            variable: The name of the environment variable to parse
            default: The default value to use if the environment variable is not set
            type: The type to cast the environment variable value to. Can be a basic type
                 (like str, int) or a complex type (list, tuple, bool)

        Raises:
            ValueError: If the environment variable value cannot be cast to the specified type
        """

        env = os.environ.get(variable)

        if env is None:
            value = default
        else:
            value = self._cast_type(env, type)

        self.dict[variable] = value

    @staticmethod
    def _cast_type(arg: str, d_type: Union[Type[T], type]) -> Any:
        """
        Cast a string argument to the specified type.

        Args:
            arg: The string value to cast
            d_type: The type to cast to. Can be a basic type (like str, int) or
                   a complex type (list, tuple, bool)

        Returns:
            The cast value

        Raises:
            ValueError: If the value cannot be cast to the specified type or
                       if the type is not supported
        """

        if isinstance(type, (list, tuple, bool)):
            try:
                cast_value = literal_eval(arg)
                return cast_value
            except (ValueError, SyntaxError):
                raise ValueError(
                    f"Argument {arg} does not match given data type or is not supported."
                )
        else:
            try:
                cast_value = d_type(arg)
                return cast_value
            except (ValueError, SyntaxError):
                raise ValueError(
                    f"Argument {arg} does not match given data type or is not supported."
                )

    def parse_args(self) -> _define_dict:
        """
        Parse all added arguments and return them in a dictionary with attribute access.

        Returns:
            A dictionary-like object that supports both dictionary access (dict['key'])
            and attribute access (dict.key) for all parsed environment variables
        """

        return self._define_dict(self.dict)


class TritonPythonModel:
    """
    A Triton inference model that processes images with GPU acceleration.

    This model handles image resizing and padding operations for inference,
    maintaining consistent GPU memory usage and optimized performance through
    pre-computed constants and GPU-based operations.
    """

    def initialize(self, args: Dict[str, Any]) -> None:
        """
        Initialize the model with configuration parameters and set up GPU constants.

        Args:
            args: Dictionary containing model initialization parameters including:
                - model_name: Name of the model
                - model_config: JSON string containing model input/output configuration
        """
        self.model_name = args["model_name"]
        model_config = json.loads(args["model_config"])
        self.inputs: List[str] = [
            input["name"] for input in model_config["input"]
        ]
        self.outputs: List[str] = [
            output["name"] for output in model_config["output"]
        ]

        parser = EnvArgumentParser()
        parser.add_arg("MODEL_DIMS", default=(640, 640), type=tuple)
        parser.add_arg("CAMERA_HEIGHT", default=720, type=int)
        parser.add_arg("CAMERA_WIDTH", default=1280, type=int)
        args = parser.parse_args()

        self.model_height: int = args.MODEL_DIMS[0]
        self.model_width: int = args.MODEL_DIMS[1]
        self.camera_height: int = args.CAMERA_HEIGHT
        self.camera_width: int = args.CAMERA_WIDTH

        self._register_constants()

    def _register_constants(self):
        """
        Pre-compute and store all constants in RAM for efficient processing.

        Calculates scaling factors, padding values, and other constants needed
        for image processing and stores them on the GPU to minimize CPU-GPU
        transfers during inference.
        """
        r = min(
            self.model_height / self.camera_height,
            self.model_width / self.camera_width,
        )

        self.newh = int(round(self.camera_height * r))
        self.neww = int(round(self.camera_width * r))

        self.top = (self.model_height - self.newh) // 2
        self.bottom = self.model_height - self.newh - self.top
        self.left = (self.model_width - self.neww) // 2
        self.right = self.model_width - self.neww - self.left

        self.pad_value = 114

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image, (self.neww, self.newh), interpolation=cv2.INTER_LINEAR
        )

        padded = cv2.copyMakeBorder(
            resized,
            self.top,
            self.bottom,
            self.left,
            self.right,
            borderType=cv2.BORDER_CONSTANT,
            value=(self.pad_value, self.pad_value, self.pad_value),
        )

        x = padded.astype(np.float32) / 255.0

        return x.transpose(2, 0, 1)[None, ...]

    def execute(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        image = pb_utils.get_input_tensor_by_name(
            requests[0], self.inputs[0]
        ).as_numpy()

        resized_image = self._resize_image(image=image)

        return [
            pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(self.outputs[0], resized_image)
                ]
            )
        ]

    def finalize(self) -> None:
        """
        Clean up resources when the model is being unloaded.
        """
        print(f"Cleaning up {self.model_name}...", flush=True)
