import os
import cv2
import typing
import numpy as np

import torch

from ast import literal_eval
from torchvision.ops import nms
from models.common import Ensemble

from urllib.parse import urlparse
from tritonclient.grpc import InferInput
from tritonclient.http import InferInput


class TritonRemoteModel:
    def __init__(self, url: str, model: str):
        parsed_url = urlparse(url)
        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient

            self.client = InferenceServerClient(parsed_url.netloc)  # Triton GRPC client
            self.model_name = model
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)

        else:
            from tritonclient.http import InferenceServerClient

            self.client = InferenceServerClient(parsed_url.netloc)  # Triton HTTP client
            self.model_name = model
            self.metadata = self.client.get_model_metadata(self.model_name)
    
    @property
    def runtime(self):
        return self.metadata.get("backend", self.metadata.get("platform"))

    def __call__(self, *args, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        inputs = self._create_inputs(*args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        result = []
        for output in self.metadata['outputs']:
            tensor = torch.as_tensor(response.as_numpy(output['name']))
            result.append(tensor)
        return result[0][0] if len(result) == 1 else result

    def _create_inputs(self, *args, **kwargs):
        args_len, kwargs_len = len(args), len(kwargs)
        if not args_len and not kwargs_len:
            raise RuntimeError("No inputs provided.")
        if args_len and kwargs_len:
            raise RuntimeError("Cannot specify args and kwargs at the same time")
        
        placeholders = [
            InferInput(i['name'], [int(s) for s in args[index].shape], i['datatype']) for index, i in enumerate(self.metadata['inputs'])
        ]
        if args_len:
            if args_len != len(placeholders):
                raise RuntimeError(f"Expected {len(placeholders)} inputs, got {args_len}.")
            for input, value in zip(placeholders, args):
                input.set_data_from_numpy(value)
        else:
            for input in placeholders:
                value = kwargs[input.name]
                input.set_data_from_numpy(value)
        return placeholders

# custom functions
class EnvArgumentParser():
    def __init__(self):
        self.dict = {}
    
    class define_dict(dict):
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
        return self.define_dict(self.dict)


# functions from YOLOv5
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


# def attempt_load(weights, device=None, inplace=True, fuse=True):
#     model = Ensemble()
#     for w in weights if isinstance(weights, list) else [weights]:
#         ckpt = torch.load(w, map_location="cpu")
#         ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()
#         model.append(ckpt.fuse().eval() if fuse else ckpt.eval())
    
#     if len(model) == 1:
#         return model[-1]
    
#     for k in "names", "nc", "yaml":
#         setattr(model, k, getattr(model[0], k))
    
#     model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride
#     assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
#     return model


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)

def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(
        prediction,
        img0_shape,
        img1_shape,
        conf_thres,
        iou_thres,
        classes=None,
        max_det=300,
        max_nms=30000,
        scale=False,
        normalize=False
):
    bs = prediction.shape[0]
    print(bs)
    xc = prediction[..., 4] > conf_thres

    # Settings
    max_nms = 30000
    redundant = True
    merge = True

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        # Apply constraints
        x = x[xc[xi]]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.half()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        boxes, scores = x[:, :4], x[:, 4]
        i = nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).half() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]

    for tensor in output:
        if scale:
            gain = min(img1_shape[0] / img0_shape[1], img1_shape[1] / img0_shape[0])
            pad = (img1_shape[1] - img0_shape[0] * gain) / 2, (img1_shape[0] - img0_shape[1] * gain) / 2

            tensor[:, [0, 2]] -= pad[0]
            tensor[:, [1, 3]] -= pad[1]
            tensor[:, :4] /= gain

            tensor[..., [0, 2]] = tensor[..., [0, 2]].clip(0, img0_shape[0])
            tensor[..., [1, 3]] = tensor[..., [1, 3]].clip(0, img0_shape[1])

        if normalize:
            tensor[:, :4] = tensor[:, :4] @ np.diag([1/img0_shape[0], 1/img0_shape[1], 1/img0_shape[0], 1/img0_shape[1]])

        tensor = tensor.numpy()

    return output