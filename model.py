import cv2
import torch
import numpy

from utilities import non_max_suppression, TritonRemoteModel



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
        img = cv2.resize(frame, self.model_dims)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = numpy.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()/255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        predictions = self.model(
			img.cpu().numpy()
		)

        preds = non_max_suppression(
            prediction=predictions,
            img0_shape=self.frame_dims,
            img1_shape=self.model_dims,
            conf_thres=self.conf,
            iou_thres=self.iou,
            classes=self.classes,
            scale=True
        )

        bboxes = [item[:4] for item in preds]
        confs = [round(float(item[4]), 2) for item in preds]
        indexes = [int(item[5]) for item in preds]

        return bboxes, confs, indexes
