import os
import cv2
import torch
import numpy
import typing
import imutils
import subprocess

from dotenv import load_dotenv
from urllib.parse import urlparse
from utilities import EnvArgumentParser



class TritonRemoteModel:
    def __init__(self, url: str, model: str):
        parsed_url = urlparse(url)
        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)
            self.model_name = model
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)
            self.config = self.client.get_model_config(self.model_name, as_json=True)["config"]

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(
                        i['name'],
                        [int(s) for s in i['shape']],
                        i['datatype']
                    )
                    for i in self.metadata['inputs']
                ]

        elif parsed_url.scheme == "http":
            from tritonclient.http import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)
            self.model_name = model
            self.metadata = self.client.get_model_metadata(self.model_name)
            self.config = self.client.get_model_config(self.model_name)

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(
                        i['name'],
                        [int(s) for s in i['shape']],
                        i['datatype']
                    )
                    for i in self.metadata['inputs']
                ]

        else:
            raise "Unsupported protocol. Use HTTP or GRPC."

        self._create_input_placeholders_fn = create_input_placeholders
        self.model_dims = self._get_dims()
        self.classes = self._get_classes()

    @property
    def runtime(self):
        return self.metadata.get("backend", self.metadata.get("platform"))

    def __call__(self, *args, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        inputs = self._create_inputs(*args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        result = []
        for output in self.metadata['outputs']:
            tensor = torch.tensor(response.as_numpy(output['name']))
            result.append(tensor)
        return result[0] if len(result) == 1 else result

    def _create_inputs(self, *args, **kwargs):
        args_len, kwargs_len = len(args), len(kwargs)
        if not args_len and not kwargs_len:
            raise RuntimeError("No inputs provided.")
        if args_len and kwargs_len:
            raise RuntimeError("Cannot specify args and kwargs at the same time")

        placeholders = self._create_input_placeholders_fn()

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

    def _get_classes(self):
        label_filename = self.config["output"][0]["label_filename"]
        docker_file_path = f"/root/app/triton/{self.model_name}/{label_filename}"
        jetson_file_path = os.path.join(os.path.abspath(os.getcwd()), f"triton/{self.model_name}/{label_filename}")

        if os.path.isfile(docker_file_path):
            with open(docker_file_path, "r") as file:
                classes = file.read().splitlines()
        elif os.path.isfile(jetson_file_path):
            with open(jetson_file_path, "r") as file:
                classes = file.read().splitlines()
        else:
            classes = None

        return classes

    def _get_dims(self):
        try:
            model_dims = tuple(self.config["input"][0]["dims"][2:4])
            return tuple(map(int, model_dims))
        except:
            return (640, 640)


class ObjectDetection():
    def __init__(
            self,
            model_name,
            triton_url
        ):

        try:
            self.model = TritonRemoteModel(url=triton_url, model=model_name)
        except ConnectionError as e:
            raise f"Failed to connect to Triton: {e}"

    def __call__(self, frame):
        predictions = self.model(
            frame
        ).tolist()

        bboxes = [item[:4] for item in predictions]
        confs = [round(float(item[4]), 2) for item in predictions]
        indexes = [int(item[5]) for item in predictions]

        return bboxes, confs, indexes


class Annotator():
    def __init__(self, classes, width=1280, height=720, santa_hat_plugin_bool=False):
        self.width = width
        self.height = height
        self.classes = classes
        self.colors = list(numpy.random.rand(len(self.classes), 3) * 255)
        self.santa_hat = cv2.imread("images/santa_hat.png")
        self.santa_hat_mask = cv2.imread("images/santa_hat_mask.png")
        self.santa_hat_plugin_bool = santa_hat_plugin_bool

    def __call__(self, frame, bboxes, confs, indexes):
        if not self.santa_hat_plugin_bool:
            for i in range(len(bboxes)):
                xmin, ymin, xmax, ymax = [int(j) for j in bboxes[i]]
                color = self.colors[indexes[i]]
                frame = cv2.rectangle(
                    img=frame,
                    pt1=(xmin, ymin),
                    pt2=(xmax, ymax),
                    color=color,
                    thickness=2
                )

                frame = cv2.putText(
                    img=frame,
                    text=f'{self.classes[indexes[i]]} ({str(confs[i])})',
                    org=(xmin, ymin - 5),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=0.75,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA
                )

            return frame

        else:
            # For santa hat plugin, turn Normalize to True in nms function
            max_index = max(range(len(confs)), key=confs.__getitem__)
            return self._overlay_obj(frame, bboxes[max_index].copy())

    def _overlay_obj(self, frame, bbox):
        bbox = [int(i * scalar) for i, scalar in zip(bbox, [self.width, self.height, self.width, self.height])]
        x, y = bbox[0], bbox[1] + 20

        resize_width = bbox[2]-bbox[0]
        santa_hat = imutils.resize(self.santa_hat.copy(), width=resize_width)
        santa_hat_mask = imutils.resize(self.santa_hat_mask.copy(), width=resize_width)
        hat_height, hat_width = santa_hat.shape[0], santa_hat.shape[1]

        mask_boolean = santa_hat_mask[:, :, 0] == 0
        mask_rgb_boolean = numpy.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)

        if x >= 0 and y >= 0:
            h = hat_height - max(0, y+hat_height-self.height)
            w = hat_width - max(0, x+hat_width-self.width)
            frame[y-h:y, x:x+w, :] = frame[y-h:y, x:x+w, :] * ~mask_rgb_boolean[0:h, 0:w, :] + (santa_hat * mask_rgb_boolean)[0:h, 0:w, :]

        elif x < 0 and y < 0:
            h = hat_height + y
            w = hat_width + x
            frame[0:0+h, 0:0+w, :] = frame[0:0+h, 0:0+w, :] * ~mask_rgb_boolean[hat_height-h:hat_height, hat_width-w:hat_width, :] + (santa_hat * mask_rgb_boolean)[hat_height-h:hat_height, hat_width-w:hat_width, :]

        elif x < 0 and y >= 0:
            h = hat_height - max(0, y+hat_height-self.height)
            w = hat_width + x
            frame[y:y+h, 0:0+w, :] = frame[y:y+h, 0:0+w, :] * ~mask_rgb_boolean[0:h, hat_width-w:hat_width, :] + (santa_hat * mask_rgb_boolean)[0:h, hat_width-w:hat_width, :]

        elif x >= 0 and y < 0:
            h = hat_height + y
            w = hat_width - max(0, x+hat_width-self.width)
            frame[0:0+h, x:x+w, :] = frame[0:0+h, x:x+w, :] * ~mask_rgb_boolean[hat_height-h:hat_height, 0:w, :] + (santa_hat * mask_rgb_boolean)[hat_height-h:hat_height, 0:w, :]

        return frame



def main(
    triton_url,
    model_name,
    stream_ip,
    stream_port,
    stream_application,
    stream_key,
    camera_index,
    camera_width,
    camera_height,
    santa_hat_plugin
):

    rtmp_url = "rtmp://{}:{}/{}/{}".format(
        stream_ip,
        stream_port,
        stream_application,
        stream_key
    )

    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    camera_fps = int(camera.get(cv2.CAP_PROP_FPS))

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

    model = ObjectDetection(
        model_name=model_name,
        triton_url=triton_url
    )

    annotator = Annotator(
        model.model.classes,
        camera_width,
        camera_height,
        santa_hat_plugin
    )

    period = 2
    tracking_index = 0

    while camera.isOpened():
        ret, frame = camera.read()

        if not ret:
            print("Frame failed to load...")
            break

        if tracking_index % period == 0:
            bboxes, confs, indexes = model(frame)
            tracking_index = 0

        if bboxes:
            frame = annotator(frame, bboxes, confs, indexes)
        tracking_index += 1

        p.stdin.write(frame.tobytes())



if __name__ == "__main__":
    load_dotenv()
    parser = EnvArgumentParser()
    parser.add_arg("TRITON_URL", default="grpc://localhost:8001", type=str)
    parser.add_arg("MODEL_NAME", default="yolov8n", type=str)
    parser.add_arg("STREAM_IP", default="127.0.0.1", type=str)
    parser.add_arg("STREAM_PORT", default=1935, type=int)
    parser.add_arg("STREAM_APPLICATION", default="live", type=str)
    parser.add_arg("STREAM_KEY", default="stream", type=str)
    parser.add_arg("CAMERA_INDEX", default=0, type=int)
    parser.add_arg("CAMERA_WIDTH", default=640, type=int)
    parser.add_arg("CAMERA_HEIGHT", default=480, type=int)
    parser.add_arg("SANTA_HAT_PLUGIN", default=False, type=bool)
    args = parser.parse_args()

    main(
        args.TRITON_URL,
        args.MODEL_NAME,
        args.STREAM_IP,
        args.STREAM_PORT,
        args.STREAM_APPLICATION,
        args.STREAM_KEY,
        args.CAMERA_INDEX,
        args.CAMERA_WIDTH,
        args.CAMERA_HEIGHT,
        args.SANTA_HAT_PLUGIN
    )
