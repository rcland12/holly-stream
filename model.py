import torch
import numpy as np

from utilities import (
    attempt_load,
    letterbox,
    non_max_suppression,
    normalize_boxes,
    scale_coords
)


class ObjectDetection():
    def __init__(self, model, img_shape=(640, 480), confidence_threshold=0.3, iou_threshold=0.1):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.fp16 = True
        else:
            self.device = torch.device("cpu")
            self.fp16 = False

        self.model = attempt_load(weights=model, device=self.device)
        self.classes = self.model.names
        self.img_shape = img_shape
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.stride = 32

    def __call__(self, frame):
        height, width = frame.shape[:2]
        img0 = frame[:, :, ::-1].copy()
        img = letterbox(img0, new_shape=self.img_shape, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.fp16 else img.float()
        img /= 255
        if len(img.shape) == 3:
            img = img[None,:,:,:]

        results = self.model(img, augment=True)[0]
        preds = non_max_suppression(results, conf_thres=self.conf, iou_thres=self.iou)[0]
        preds[:, :4] = scale_coords(img.shape[2:], preds[:, :4], img0.shape).round()

        height, width = img0.shape[:2]
        bboxes = [normalize_boxes(item[:4], width, height) for item in preds]
        conf = [float(item[4]) for item in preds]
        obj = [int(item[5]) for item in preds]
        names = [self.classes[item] for item in obj]
        predictions = []

        if bboxes is not None and len(bboxes):
            for i in range(len(bboxes)):
                prediction = {
                    "bbox": bboxes[i],
                    "conf": conf[i],
                    "name": names[i],
                }
                predictions.append(prediction)

        return predictions
