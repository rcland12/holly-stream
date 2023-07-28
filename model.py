import cv2
import torch
import numpy

from utilities import (
    attempt_load,
    non_max_suppression
)


class ObjectDetection():
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = attempt_load(weights=model_path, device=self.device)
        self.classes = self.model.names
        self.new_width = 320
        self.stride = 32

    def __call__(self, frame, confidence_threshold=0.3, iou_threshold=0.45):
        height, width = frame.shape[:2]
        new_height = int((((self.new_width / width) * height) // self.stride) * self.stride)

        img = cv2.resize(frame, (self.new_width, new_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = numpy.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()/255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        preds = self.model(img, augment=False)[0]
        preds = non_max_suppression(
            preds,
            conf_thres=confidence_threshold,
            iou_thres=iou_threshold,
            classes=16
        )
        predictions = []
        
        if preds[0] is not None and len(preds):
            for pred in preds[0]:
                label = self.classes[int(pred[5])]
                xmin = int(pred[0] * width / self.new_width)
                ymin = int(pred[1] * height / new_height)
                xmax = int(pred[2] * width / self.new_width)
                ymax = int(pred[3] * height / new_height)

                prediction = {
                    'label': label,
                    'bbox' : [xmin, ymin, xmax, ymax]
                }

                predictions.append(prediction)

        return predictions


# from utilities import (
#     attempt_load,
#     letterbox,
#     non_max_suppression,
#     normalize_boxes,
#     scale_coords
# )

# class ObjectDetection():
#     def __init__(self, model, img_shape=(640, 640), confidence_threshold=0.3, iou_threshold=0.1):
#         if torch.cuda.is_available():
#             self.device = "cuda"
#             self.fp16 = False
#         else:
#             self.device = "cpu"
#             self.fp16 = False

#         self.model = attempt_load(weights=model, device=self.device)
#         self.classes = self.model.names
#         self.img_shape = img_shape
#         self.conf = confidence_threshold
#         self.iou = iou_threshold
#         self.stride = 32

#     def __call__(self, frame):
#         img0 = frame[:, :, ::-1].copy()
#         img = letterbox(img0, new_shape=self.img_shape, stride=self.stride, auto=True)[0]
#         img = img.transpose((2, 0, 1))[::-1]
#         img = numpy.ascontiguousarray(img)
#         img = torch.from_numpy(img).to(self.device)
#         img = img.half() if self.fp16 else img.float()
#         img /= 255
#         if len(img.shape) == 3:
#             img = img[None,:,:,:]

#         results = self.model(img, augment=True)[0]
#         preds = non_max_suppression(results, conf_thres=self.conf, iou_thres=self.iou)[0]
#         preds[:, :4] = scale_coords(img.shape[2:], preds[:, :4], img0.shape).round()

#         height, width = img0.shape[:2]
#         bboxes = [normalize_boxes(item[:4], width, height) for item in preds]
#         conf = [float(item[4]) for item in preds]
#         obj = [int(item[5]) for item in preds]
#         names = [self.classes[item] for item in obj]
#         predictions = []

#         if bboxes is not None and len(bboxes):
#             for i in range(len(bboxes)):
#                 prediction = {
#                     "bbox": bboxes[i],
#                     "score": conf[i],
#                     "label": names[i],
#                 }
#                 predictions.append(prediction)

#         return predictions
