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

model = TritonRemoteModel("http://localhost:8000", "yolov5n")

frame = cv2.imread("dogs.jpg")

response = model(frame)

print(response)