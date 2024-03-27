import os
import cv2
import json
import numpy as np
import triton_python_backend_utils as pb_utils

from ast import literal_eval
from dotenv import load_dotenv
from typing import Any, Dict, List, Tuple, Optional, Type
from c_python_backend_utils import InferenceRequest, InferenceResponse



class EnvArgumentParser:
    def __init__(self):
        self.dict = {}

    class _define_dict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def add_arg(self, variable, default=None, type=str):
        env = os.environ.get(variable)

        if env is None:
            value = default
        else:
            value = self._cast_type(env, type)

        self.dict[variable] = value

    @staticmethod
    def _cast_type(arg, d_type):
        if d_type == list or d_type == tuple or d_type == bool:
            try:
                cast_value = literal_eval(arg)
                return cast_value
            except (ValueError, SyntaxError):
                raise ValueError(f"Argument {arg} does not match given data type or is not supported.")
        else:
            try:
                cast_value = d_type(arg)
                return cast_value
            except (ValueError, SyntaxError):
                raise ValueError(f"Argument {arg} does not match given data type or is not supported.")
    
    def parse_args(self):
        return self._define_dict(self.dict)


def letterbox(
    image: np.ndarray = None,
    new_shape: Tuple[int, int] = (640, 640),
    output_type: str = 'float32'
) -> np.ndarray:
    shape = image.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    return image.transpose((2, 0, 1))[::-1].astype(output_type)


class TritonPythonModel:
    def initialize(self, args: Dict[str, Any]) -> None:
        model_config = json.loads(args["model_config"])
        OUTPUT_0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT_0")
    
        load_dotenv()
        parser = EnvArgumentParser()
        parser.add_arg("MODEL_DIMS", default=(640, 640), type=tuple)
        args = parser.parse_args()

        self.model_dims = args.MODEL_DIMS
        self.output_type = pb_utils.triton_string_to_numpy(OUTPUT_0_config["data_type"])

    def execute(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        responses = []
        for request in requests:
            image = letterbox(
                image=pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy(),
                new_shape=self.model_dims,
                output_type=self.output_type
            )
            image /= 255

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "OUTPUT_0",
                            image[None]
                        )
                    ]
                )
            )
          
        return responses

    def finalize(self) -> None:
        print('Cleaning up preprocess model...')