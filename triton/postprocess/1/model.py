import json
import torch
import numpy as np

from torchvision.ops import nms
from ensemble_boxes import weighted_boxes_fusion

import triton_python_backend_utils as pb_utils


# def xywh2xyxy(x):
#     y = x.copy()
#     y[..., 0] = x[..., 0] - x[..., 2] / 2
#     y[..., 1] = x[..., 1] - x[..., 3] / 2
#     y[..., 2] = x[..., 0] + x[..., 2] / 2
#     y[..., 3] = x[..., 1] + x[..., 3] / 2
#     return y

# def box_area(box):
#     return (box[2] - box[0]) * (box[3] - box[1])

# def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray, eps=1e-7) -> np.ndarray:
#     area_a = box_area(boxes_a.T)
#     area_b = box_area(boxes_b.T)

#     top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
#     bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

#     area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
        
#     return area_inter / (area_a[:, None] + area_b - area_inter)

# def nms(predictions: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
#     rows, _ = predictions.shape

#     sort_index = np.flip(predictions[:, 4].argsort())
#     predictions = predictions[sort_index, :]

#     boxes = predictions[:, :4]
#     categories = predictions[:, 5]
#     ious = box_iou_batch(boxes, boxes)
#     ious = ious - np.eye(rows)

#     keep = np.ones(rows, dtype=bool)
#     ious_mask = (ious > iou_threshold) & (categories == categories[:, None])
#     keep_mask = np.logical_not(np.triu(ious_mask))
#     keep &= np.all(keep_mask, axis=1)

#     return keep

# def non_max_suppression(
#         prediction,
#         img0_shape,
#         img1_shape,
#         conf_thres,
#         iou_thres,
#         scale=False,
#         normalize=False
# ):
#     boxes = xywh2xyxy(prediction[prediction[:, 4].reshape(-1) > conf_thres, :])
#     output = boxes[nms(boxes, iou_thres), :]

#     if scale:
#         gain = min(img1_shape[0] / img0_shape[1], img1_shape[1] / img0_shape[0])
#         pad = (img1_shape[1] - img0_shape[0] * gain) / 2, (img1_shape[0] - img0_shape[1] * gain) / 2

#         output[:, [0, 2]] -= pad[0]
#         output[:, [1, 3]] -= pad[1]
#         output[:, :4] /= gain

#         output[..., [0, 2]] = output[..., [0, 2]].clip(0, img0_shape[0])
#         output[..., [1, 3]] = output[..., [1, 3]].clip(0, img0_shape[1])

#     if normalize:
#         output[:, :4] = output[:, :4] @ np.diag([1/img0_shape[0], 1/img0_shape[1], 1/img0_shape[0], 1/img0_shape[1]])

#     return output  


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
        img0_shape=(640, 480),
        img1_shape=(640, 640),
        conf_thres=0.5,
        iou_thres=0.4,
        max_det=300,
        max_nms=30000,
        scale=True,
        normalize=True
):
    bs = prediction.shape[0]
    xc = prediction[..., 4] > conf_thres

    # Settings
    max_nms = 30000
    redundant = True
    merge = True

    output = [torch.zeros((0, 6))] * bs
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

        if scale:
            gain = min(img1_shape[0] / img0_shape[1], img1_shape[1] / img0_shape[0])
            pad = (img1_shape[1] - img0_shape[0] * gain) / 2, (img1_shape[0] - img0_shape[1] * gain) / 2

            x[:, [0, 2]] -= pad[0]
            x[:, [1, 3]] -= pad[1]
            x[:, :4] /= gain

            x[..., 0].clamp_(0, img0_shape[0])
            x[..., 1].clamp_(0, img0_shape[1])
            x[..., 2].clamp_(0, img0_shape[0])
            x[..., 3].clamp_(0, img0_shape[1])

        if normalize:
            x[..., :4] = torch.mm(
                x[..., :4],
                torch.diag(torch.Tensor([1/img0_shape[0], 1/img0_shape[1], 1/img0_shape[0], 1/img0_shape[1]]))
            )

        output[xi] = x[i].numpy()

    return output    



class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        self.num_inputs = len(model_config["input"]) # (hw, nhw, conf_iou, class0, class1, ...)

    def execute(self, requests):
        responses = []
        for request in requests:
            width, height = pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy()

            results = non_max_suppression(
                torch.tensor([
                    pb_utils.get_input_tensor_by_name(request, "INPUT_{}".format(model)).as_numpy()
                    for model in range(3, self.num_inputs)
                ])
            )
            
            boxes, scores, labels = weighted_boxes_fusion(
                [result[:, :4] for result in results],
                [result[:, 4] for result in results],
                [result[:, 5] for result in results]
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("OUTPUT_0", boxes),
                    pb_utils.Tensor("OUTPUT_1", scores),
                    pb_utils.Tensor("OUTPUT_2", labels)
                ]
            )
            responses.append(inference_response)
          
        return responses

    def finalize(self):
        
        print('Cleaning up...')