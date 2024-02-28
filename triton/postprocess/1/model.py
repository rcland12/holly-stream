import os
import torch
import numpy as np
from torch.utils.dlpack import from_dlpack
import triton_python_backend_utils as pb_utils

from ast import literal_eval
from dotenv import load_dotenv
from torchvision.ops import nms



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


def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    dw = x[..., 2] / 2
    dh = x[..., 3] / 2
    y[..., 0] = x[..., 0] - dw
    y[..., 1] = x[..., 1] - dh
    y[..., 2] = x[..., 0] + dw
    y[..., 3] = x[..., 1] + dh
    return y


def non_max_suppression(
    prediction,
    img0_shape=(640, 480),
    img1_shape=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    scale=True,
    normalize=False
):
    bs = prediction.shape[0]
    nc = prediction.shape[1] - 4
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    prediction = prediction.transpose(-1, -2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)

        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > 30000:
            x = x[x[:, 4].argsort(descending=True)[:30000]]

        c = x[:, 5:6] * 7680
        scores = x[:, 4]

        boxes = x[:, :4] + c
        i = nms(boxes, scores, iou_thres)
        i = i[:300]

        output[xi] = x[i]

    output = output[0]

    if scale:
        gain = min(img1_shape[0] / img0_shape[1], img1_shape[1] / img0_shape[0])
        pad = (img1_shape[1] - img0_shape[0] * gain) / 2, (img1_shape[0] - img0_shape[1] * gain) / 2

        output[:, [0, 2]] -= pad[0]
        output[:, [1, 3]] -= pad[1]
        output[:, :4] /= gain

        output[..., 0].clamp_(0, img0_shape[0])
        output[..., 1].clamp_(0, img0_shape[1])
        output[..., 2].clamp_(0, img0_shape[0])
        output[..., 3].clamp_(0, img0_shape[1])

    if normalize:
        output[..., :4] = torch.mm(
            output[..., :4],
            torch.diag(torch.Tensor([1/img0_shape[0], 1/img0_shape[1], 1/img0_shape[0], 1/img0_shape[1]]))
        )

    return output.numpy()



class TritonPythonModel:
    def initialize(self, args):
        load_dotenv()
        parser = EnvArgumentParser()
        parser.add_arg("CAMERA_WIDTH", default=640, type=int)
        parser.add_arg("CAMERA_HEIGHT", default=480, type=int)
        parser.add_arg("MODEL_DIMS", default=(640, 640), type=tuple)
        parser.add_arg("CONFIDENCE_THRESHOLD", default=0.3, type=float)
        parser.add_arg("IOU_THRESHOLD", default=0.25, type=float)
        parser.add_arg("CLASSES", default=None, type=list)
        parser.add_arg("SANTA_HAT_PLUGIN", default=False, type=bool)
        args = parser.parse_args()

        self.camera_width = args.CAMERA_WIDTH
        self.camera_height = args.CAMERA_HEIGHT
        self.model_dims = args.MODEL_DIMS
        self.conf_thres = args.CONFIDENCE_THRESHOLD
        self.iou_thres = args.IOU_THRESHOLD
        self.classes = args.CLASSES
        self.santa_hat_plugin = args.SANTA_HAT_PLUGIN
 
    def execute(self, requests):
        responses = []
        for request in requests:
            results = non_max_suppression(
                from_dlpack(pb_utils.get_input_tensor_by_name(request, "INPUT_0").to_dlpack()),
                img0_shape=(self.camera_width, self.camera_height),
                img1_shape=self.model_dims,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                classes=self.classes,
                normalize=self.santa_hat_plugin
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("OUTPUT_0", results)
                    ]
                )
            )

        return responses

    def finalize(self):
        print('Cleaning up postprocess model...')