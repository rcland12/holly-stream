import cv2
import torch
import numpy

# from utilities import (
#     attempt_load,
#     non_max_suppression
# )


# class ObjectDetection():
#     def __init__(self, model_path):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = attempt_load(weights=model_path, device=self.device)
#         self.classes = self.model.names
#         self.new_width = 320
#         self.stride = 32

#     def __call__(self, frame, classes=None, confidence_threshold=0.3, iou_threshold=0.45):
#         height, width = frame.shape[:2]
#         new_height = int((((self.new_width / width) * height) // self.stride) * self.stride)

#         img = cv2.resize(frame, (self.new_width, new_height))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = numpy.moveaxis(img, -1, 0)
#         img = torch.from_numpy(img).to(self.device)
#         img = img.float()/255.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)

#         preds = self.model(img, augment=False)[0]
#         preds = non_max_suppression(
#             preds,
#             conf_thres=confidence_threshold,
#             iou_thres=iou_threshold,
#             classes=classes,
#             nc=len(self.classes)
#         )
#         predictions = []
        
#         if preds[0] is not None and len(preds):
#             for pred in preds[0]:
#                 score = numpy.round(pred[4].cpu().detach().numpy(), 2)
#                 label = self.classes[int(pred[5])]
#                 xmin = int(pred[0] * width / self.new_width)
#                 ymin = int(pred[1] * height / new_height)
#                 xmax = int(pred[2] * width / self.new_width)
#                 ymax = int(pred[3] * height / new_height)

#                 prediction = {
#                     'bbox' : [xmin, ymin, xmax, ymax],
#                     'score': score,
#                     'label': label
#                 }

#                 predictions.append(prediction)

#         return predictions


from utilities import (
    letterbox,
    non_max_suppression,
    scale_coords,
    TritonRemoteModel
)


class ObjectDetection():
    def __init__(self, model_name, classes):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = attempt_load(weights=model, device=self.device)
        self.model = TritonRemoteModel(url="http://localhost:8000", model=model_name)
        self.classes = classes

    def __call__(self, frame, classes=None, confidence_threshold=0.3, iou_threshold=0.45):
        img0 = frame[:, :, ::-1].copy()
        img = letterbox(img0, auto=False)
        img = img.transpose((2, 0, 1))[::-1]
        img = numpy.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255
        if len(img.shape) == 3:
            img = img[None,:,:,:]

        preds = self.model(
			img.cpu().numpy()
		)
        print(preds.shape)
        preds = non_max_suppression(
            preds,
            img0.shape,
            img.shape,
            conf_thres=confidence_threshold,
            iou_thres=iou_threshold,
            classes=None,
            scale=True,
            normalize=False
        )[0]
        print(preds)

        height, width = img0.shape[:2]
        bboxes = [item[:4] for item in preds]
        conf = [round(float(item[4]), 2) for item in preds]
        obj = [int(item[5]) for item in preds]
        names = [self.classes[item] for item in obj]
        predictions = []

        if bboxes is not None and len(bboxes):
            for i in range(len(bboxes)):
                prediction = {
                    "bbox": bboxes[i],
                    "score": conf[i],
                    "label": names[i],
                }
                predictions.append(prediction)

        return predictions
