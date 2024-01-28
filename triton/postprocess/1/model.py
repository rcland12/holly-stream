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
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def non_max_suppression(
    prediction,
    img0_shape=(640, 480),
    img1_shape=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,
    max_nms=30000,
    max_wh=7680,
    scale=True
):
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    multi_label &= nc > 1

    prediction = prediction.transpose(-1, -2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh
        scores = x[:, 4]

        boxes = x[:, :4] + c
        i = nms(boxes, scores, iou_thres)
        i = i[:max_det]

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
        args = parser.parse_args()

        self.camera_width = args.CAMERA_WIDTH
        self.camera_height = args.CAMERA_HEIGHT
        self.model_dims = args.MODEL_DIMS
        self.conf_thres = args.CONFIDENCE_THRESHOLD
        self.iou_thres = args.IOU_THRESHOLD
        self.classes = args.CLASSES
 
    def execute(self, requests):
        responses = []
        for request in requests:
            results = non_max_suppression(
                from_dlpack(pb_utils.get_input_tensor_by_name(request, "INPUT_0").to_dlpack()),
                img0_shape=(self.camera_width, self.camera_height),
                img1_shape=self.model_dims,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                classes=self.classes
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("OUTPUT_0", results),
                    ]
                )
            )
          
        return responses

    def finalize(self):
        print('Cleaning up postprocess model...')