import cv2
import torch
import numpy
import typing

from torchvision.ops import nms
from urllib.parse import urlparse
from tritonclient.grpc import InferInput
from tritonclient.http import InferInput



class TritonRemoteModel:
    def __init__(self, url: str, model: str):
        parsed_url = urlparse(url)
        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient

        elif parsed_url.scheme == "http":
            from tritonclient.http import InferenceServerClient
        
        else:
            raise "Unsupported protocol. Use HTTP or GRPC."

        self.client = InferenceServerClient(parsed_url.netloc)  # Triton GRPC client
        self.model_name = model
        self.metadata = self.client.get_model_metadata(self.model_name)
        self.config = self.client.get_model_config(self.model_name)
        try:
            self.model_dims = self.config["config"]["input"][0]["dims"][2:4]
        except:
            self.model_dims = (640, 640)
    
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


def preprocess_frame(frame, model_dims, device):
    frame = cv2.resize(frame, model_dims)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = numpy.moveaxis(frame, -1, 0)
    frame = torch.from_numpy(frame).to(device)
    frame = frame.float()/255.0
    if frame.ndimension() == 3:
        frame = frame.unsqueeze(0)
    return frame


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


def postprocess(
        predictions,
        img0_shape,
        img1_shape,
        conf_thres,
        iou_thres,
        classes=None,
        max_det=300,
        max_nms=30000,
        scale=False
):
    predictions = predictions[None,:,:]
    xc = predictions[..., 4] > conf_thres

    # Settings
    max_nms = 30000
    redundant = True
    merge = True

    output = [torch.zeros((0, 6), device=predictions.device)]
    for xi, x in enumerate(predictions):
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

        tensor = tensor.numpy()

    return output[0]


class ObjectDetection():
    def __init__(
            self,
            model_name,
            all_classes,
            classes=None,
            camera_width=640,
            camera_height=480,
            confidence_threshold=0.3,
            iou_threshold=0.25,
            triton_url="http://localhost:8000"
        ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TritonRemoteModel(url=triton_url, model=model_name)
        self.all_classes = all_classes
        self.classes = classes
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.frame_dims = (camera_width, camera_height)
        self.model_dims = self.model.model_dims


    def __call__(self, frame):
        processed_frame = preprocess_frame(
            frame=frame,
            model_dims=self.model_dims,
            device=self.device
        )

        predictions = self.model(
			processed_frame.cpu().numpy()
		)

        predictions = postprocess(
            predictions=predictions,
            img0_shape=self.frame_dims,
            img1_shape=self.model_dims,
            conf_thres=self.conf,
            iou_thres=self.iou,
            classes=self.classes,
            scale=True
        )

        bboxes = [item[:4] for item in predictions]
        confs = [round(float(item[4]), 2) for item in predictions]
        indexes = [int(item[5]) for item in predictions]

        return bboxes, confs, indexes
