import os
import torch
import numpy as np
import triton_python_backend_utils as pb_utils

from ast import literal_eval
from dotenv import load_dotenv
from torchvision.ops import nms
from torch.utils.dlpack import from_dlpack
from typing import Any, Dict, List, Tuple, Optional, Type



class EnvArgumentParser:
    def __init__(self):
        self.dict: Dict[str, Any] = {}

    class _define_dict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def add_arg(self, variable: str, default: Any = None, type: Type = str) -> None:
        env = os.environ.get(variable)

        if env is None:
            value = default
        else:
            value = self._cast_type(env, type)

        self.dict[variable] = value

    @staticmethod
    def _cast_type(arg: str, d_type: Type) -> Any:
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
    
    def parse_args(self) -> '_define_dict':
        return self._define_dict(self.dict)


def box_area(box: torch.Tensor) -> torch.Tensor:
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(
        prediction: torch.Tensor,
        img0_shape: Tuple[int, int] = (1280, 720),
        img1_shape: Tuple[int, int] = (640, 640),
        conf_thres: float = 0.3,
        iou_thres: float = 0.25,
        classes: Optional[List[int]] = None,
        scale: bool = True,
        normalize: bool = False
) -> np.ndarray:
    prediction = prediction.float()
    bs = prediction.shape[0]
    xc = prediction[..., 4] > conf_thres

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
        elif n > 30000:
            x = x[x[:, 4].argsort(descending=True)[:30000]]

        boxes, scores = x[:, :4], x[:, 4]
        i = nms(boxes, scores, iou_thres)

        if i.shape[0] > 300:
            i = i[:300]

        iou = box_iou(boxes[i], boxes) > iou_thres
        weights = iou * scores[None]
        x[i, :4] = torch.mm(weights, x[:, :4]).half() / weights.sum(1, keepdim=True)
        i = i[iou.sum(1) > 1]

        output[xi] = x[i]

    output = output[0]

    if scale:
        gain = min(img1_shape[0] / img0_shape[1], img1_shape[1] / img0_shape[0])
        pad = (img1_shape[1] - img0_shape[0] * gain) / 2, (img1_shape[0] - img0_shape[1] * gain) / 2

        output[:, [0, 2]] -= pad[0]
        output[:, [1, 3]] -= pad[1]
        output[:, :4] /= gain

        output[..., [0, 2]] = output[..., [0, 2]].clip(0, img0_shape[0])
        output[..., [1, 3]] = output[..., [1, 3]].clip(0, img0_shape[1])

    if normalize:
        output[..., :4] = torch.mm(
            output[..., :4],
            torch.diag(torch.Tensor([1/img0_shape[0], 1/img0_shape[1], 1/img0_shape[0], 1/img0_shape[1]]))
        )

    return output.numpy()


class TritonPythonModel:
    def initialize(self, args: Dict[str, Any]) -> None:
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
 
    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
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

    def finalize(self) -> None:
        print('Cleaning up postprocess model...')
