import os
import torch
import numpy as np
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
        img1_shape=(640, 640),
        conf_thres=0.3,
        iou_thres=0.25,
        classes=None,
        max_det=300,
        max_nms=30000,
        scale=True,
        normalize=True
):
    bs = prediction.shape[0]
    xc = prediction[..., 4] > conf_thres

    max_nms = 30000
    redundant = True
    merge = True

    output = [torch.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.half()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

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

    return np.concatenate(output, axis=0)



class TritonPythonModel:
    def initialize(self, args):
        load_dotenv()
        parser = EnvArgumentParser()
        parser.add_arg("CLASSES", default=None, type=list)
        parser.add_arg("MODEL_DIMS", default=(640, 640), type=tuple)
        parser.add_arg("CONFIDENCE_THRESHOLD", default=0.3, type=float)
        parser.add_arg("IOU_THRESHOLD", default=0.25, type=float)
        args = parser.parse_args()

        self.classes = args.CLASSES
        self.model_dims = args.MODEL_DIMS
        self.conf_thres = args.CONFIDENCE_THRESHOLD
        self.iou_thres = args.IOU_THRESHOLD
 
    def execute(self, requests):
        responses = []
        for request in requests:
            results = non_max_suppression(
                torch.tensor(
                    pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy()
                ),
                pb_utils.get_input_tensor_by_name(request, "INPUT_1").as_numpy(),
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
        print('Cleaning up...')