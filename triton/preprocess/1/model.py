import os
import cv2
import numpy
import triton_python_backend_utils as pb_utils

from ast import literal_eval
from dotenv import load_dotenv



class EnvArgumentParser():
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
            value = self.cast_type(env, type)

        self.dict[variable] = value

    def cast_type(self, arg, d_type):
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
    image=None,
    new_shape=(640, 640),
    auto=False,
    scaleFill=False,
    scaleup=True,
    center=True,
    stride=32,
    labels=None
):
    labels = {}
    img = image
    shape = img.shape[:2]  # current shape [height, width]
    new_shape = labels.pop("rect_shape", self.new_shape)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if self.auto:  # minimum rectangle
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
    elif self.scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    if self.center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    if labels.get("ratio_pad"):
        labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

    return img



class TritonPythonModel:
    def initialize(self, args):
        load_dotenv()
        parser = EnvArgumentParser()
        parser.add_arg("MODEL_DIMS", default=(640, 640), type=tuple)
        args = parser.parse_args()

        self.model_dims = args.MODEL_DIMS

    def execute(self, requests):
        responses = []
        for request in requests:
            img = letterbox(
                pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy(),
                self.model_dims
            )
            img = np.stack(img)
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            img = np.ascontiguousarray(img).astype('float16')  # contiguous
            img /= 255  # 0 - 255 to 0.0 - 1.0

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "OUTPUT_0",
                            img[None]
                        )
                    ]
                )
            )
          
        return responses

    def finalize(self):
        print('Cleaning up preprocess model...')


# model_warmup [{
#     name : "preprocess model warmup"
#     batch_size: 1
#     inputs {
#       key: "INPUT_0"
#       value: {
#         data_type: TYPE_UINT8
#         dims: 1280
#         dims: 720
#         dims: 3
#         input_data_file: "INPUT_0"
#       }
#     }
# }]