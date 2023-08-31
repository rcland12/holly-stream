import cv2
import torch
import numpy

from utilities import (
    letterbox,
    non_max_suppression,
    TritonRemoteModel
)


class ObjectDetection():
    def __init__(self, model_name, classes):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TritonRemoteModel(url="http://localhost:8000", model=model_name)
        self.classes = classes

    def __call__(self, frame, classes=None, confidence_threshold=0.3, iou_threshold=0.45):
        # img0 = frame[:, :, ::-1].copy()
        # img = letterbox(img0, auto=False)
        # img = img.transpose((2, 0, 1))[::-1]
        # img = numpy.ascontiguousarray(img)
        # img = torch.from_numpy(img).to(self.device)
        # img = img.float()
        # img /= 255
        # if len(img.shape) == 3:
        #     img = img[None,:,:,:]

        height, width = frame.shape[:2]
        new_height = int((((640 / width) * height) // 32) * 32)
        img = cv2.resize(frame, (640, new_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = numpy.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()/255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        preds = self.model(
			img
		)
        preds = preds[None,:,:]
        preds = non_max_suppression(
            preds,
            (width, height),
            (640, 640),
            conf_thres=confidence_threshold,
            iou_thres=iou_threshold,
            classes=None,
            scale=True,
            normalize=False
        )

        bboxes = [item[:4] for item in preds]
        confs = [round(float(item[4]), 2) for item in preds]
        obj = [int(item[5]) for item in preds]
        names = [self.classes[item] for item in obj]

        return bboxes, confs, names

        # if bboxes is not None and len(bboxes):
        #     for i in range(len(bboxes)):
        #         prediction = {
        #             "bbox": bboxes[i],
        #             "score": conf[i],
        #             "label": names[i],
        #         }
        #         predictions.append(prediction)

        # return predictions
