import os
import cv2
import torch
import numpy
import typing
import subprocess

from nanocamera import Camera
from dotenv import load_dotenv
from torchvision.ops import nms
from urllib.parse import urlparse
from utilities import EnvArgumentParser
from tritonclient.grpc import InferInput
from tritonclient.http import InferInput



class TritonRemoteModel:
    def __init__(self, url: str, model: str):
        parsed_url = urlparse(url)
        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient

            self.client = InferenceServerClient(parsed_url.netloc)
            self.model_name = model
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)
            self.config = self.client.get_model_config(self.model_name, as_json=True)["config"]

        elif parsed_url.scheme == "http":
            from tritonclient.http import InferenceServerClient

            self.client = InferenceServerClient(parsed_url.netloc)
            self.model_name = model
            self.metadata = self.client.get_model_metadata(self.model_name)
            self.config = self.client.get_model_config(self.model_name)

        else:
            raise "Unsupported protocol. Use HTTP or GRPC."

        try:
            model_dims = tuple(self.config["input"][0]["dims"][2:4])
            self.model_dims = tuple(map(int, model_dims))

            label_filename = self.config["output"][0]["label_filename"]
            docker_file_path = f"/root/app/triton/{model}/{label_filename}"
            jetson_file_path = os.path.join(os.path.abspath(os.getcwd()), f"triton/{model}/{label_filename}")
            if os.path.isfile(docker_file_path):
                with open(docker_file_path, "r") as file:
                    self.classes = file.read().splitlines()
            elif os.path.isfile(jetson_file_path):
                with open(jetson_file_path, "r") as file:
                    self.classes = file.read().splitlines()
            else:
                raise "Class labels file is invalid or is in the wrong location."
        except:
            self.model_dims = (640, 640)
            self.classes = None

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
            camera_width,
            camera_height,
            triton_url
        ):
 
        try:
            self.model = TritonRemoteModel(url=triton_url, model=model_name)
        except ConnectionError as e:
            raise f"Failed to connect to Triton: {e}"

        self.frame_dims = [camera_width, camera_height]

    def __call__(self, frame):
        predictions = self.model(
            frame,
            numpy.array(self.frame_dims, dtype='int16')
        )[0]
        print(predictions)
        bboxes = [item[:4] for item in predictions]
        confs = [round(float(item[4]), 2) for item in predictions]
        indexes = [int(item[5]) for item in predictions]

        return bboxes, confs, indexes


def main(
    object_detection,
    triton_url,
    model_name,
    classes,
    confidence_threshold,
    iou_threshold,
    stream_ip,
    stream_port,
    stream_application,
    stream_key,
    camera_index,
    camera_width,
    camera_height,
    camera_fps
):

    rtmp_url = "rtmp://{}:{}/{}/{}".format(
        stream_ip,
        stream_port,
        stream_application,
        stream_key
    )

    camera = Camera(
        device_id=camera_index,
        flip=0,
        width=camera_width,
        height=camera_height,
        fps=camera_fps
    )

    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', "{}x{}".format(camera_width, camera_height),
        '-r', str(camera_fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv',
        rtmp_url
    ]

    p = subprocess.Popen(command, stdin=subprocess.PIPE)

    if object_detection:
        model = ObjectDetection(
            model_name=model_name,
            camera_width=camera_width,
            camera_height=camera_height,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            triton_url=triton_url,
            classes=classes
        )

        colors = list(numpy.random.rand(len(model.model.classes), 3) * 255)

        while camera.isReady():
            frame = camera.read()

            bboxes, confs, indexes = model(frame)

            for i in range(len(bboxes)):
                xmin, ymin, xmax, ymax = bboxes[i]
                color = colors[indexes[i]]
                frame = cv2.rectangle(
                    img=frame,
                    pt1=(xmin, ymin),
                    pt2=(xmax, ymax),
                    color=color,
                    thickness=2
                )
            frame = cv2.putText(
                img=frame,
                text=f'{model.model.classes[indexes[i]]} ({str(confs[i])})',
                org=(xmin, ymin),
                fontFace=cv2.FONT_HERSHEY_PLAIN ,
                fontScale=0.75,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA
            )

            p.stdin.write(frame.tobytes())

    else:
        while camera.isReady():
            frame = camera.read()
            p.stdin.write(frame.tobytes())

    camera.release()
    del camera


if __name__ == "__main__":
    load_dotenv()
    parser = EnvArgumentParser()
    parser.add_arg("OBJECT_DETECTION", default=True, type=bool)
    parser.add_arg("TRITON_URL", default="http://localhost:8000/", type=str)
    parser.add_arg("MODEL", default="yolov5n", type=str)
    parser.add_arg("CLASSES", default=None, type=list)
    parser.add_arg("CONFIDENCE_THRESHOLD", default=0.3, type=float)
    parser.add_arg("IOU_THRESHOLD", default=0.45, type=float)
    parser.add_arg("STREAM_IP", default="127.0.0.1", type=str)
    parser.add_arg("STREAM_PORT", default=1935, type=int)
    parser.add_arg("STREAM_APPLICATION", default="live", type=str)
    parser.add_arg("STREAM_KEY", default="stream", type=str)
    parser.add_arg("CAMERA_INDEX", default=0, type=int)
    parser.add_arg("CAMERA_WIDTH", default=640, type=int)
    parser.add_arg("CAMERA_HEIGHT", default=480, type=int)
    parser.add_arg("CAMERA_FPS", default=30, type=int)
    args = parser.parse_args()

    main(
        args.OBJECT_DETECTION,
        args.TRITON_URL,
        args.MODEL,
        args.CLASSES,
        args.CONFIDENCE_THRESHOLD,
        args.IOU_THRESHOLD,
        args.STREAM_IP,
        args.STREAM_PORT,
        args.STREAM_APPLICATION,
        args.STREAM_KEY,
        args.CAMERA_INDEX,
        args.CAMERA_WIDTH,
        args.CAMERA_HEIGHT,
        args.CAMERA_FPS
    )


# output [
#   {
#     name: "output"
#     data_type: TYPE_FP32
#     dims: [ -1, -1, -1 ]
#     label_filename: "yolov5_labels.txt"
#   },
#   {
#     name: "345"
#     data_type: TYPE_FP16
#     dims: [ -1, -1, -1, -1, -1 ]
#   },
#   {
#     name: "405"
#     data_type: TYPE_FP16
#     dims: [ -1, -1, -1, -1, -1 ]
#   },
#   {
#     name: "465"
#     data_type: TYPE_FP16
#     dims: [ -1, -1, -1, -1, -1 ]
#   }
# ]

# preprocess warmup
# model_warmup [{
#     name : "preprocess model warmup"
#     batch_size: 1
#     count: 1
#     inputs {
#       key: "INPUT_0"
#       value: {
#         data_type: TYPE_UINT8
#         dims: 1
#         dims: 1015
#         dims: 671
#         dims: 3
#         input_data_file: "input_0"
#       }
#     }
# }]

# postprocess warmup
# model_warmup [
#   {
#     name : "postprocess model warmup"
#     batch_size: 1
#     count: 1
#     inputs {
#       key: "INPUT_0"
#       value: {
#           data_type: TYPE_INT16
#           dims: 2
#           input_data_file: "input_0"
#       }
#     }
#     inputs {
#       key: "INPUT_1"
#       value: {
#           data_type: TYPE_INT16
#           dims: 2
#           input_data_file: "input_1"
#       }
#     }
#     inputs {
#       key: "INPUT_2"
#       value: {
#           data_type: TYPE_FP16
#           dims: 2
#           input_data_file: "input_2"
#       }
#     }
#     inputs {
#       key: "INPUT_3"
#       value: {
#           data_type: TYPE_FP32
#           dims: 1
#           dims: 25200
#           dims: 6
#           input_data_file: "input_3"
#       }
#     }
#   }
# ]