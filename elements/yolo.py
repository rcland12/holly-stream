import torch
import cv2
import numpy as np

from utilities import attempt_load
from utilities import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OBJ_DETECTION():
    def __init__(self, model_path, classes):
        self.classes = classes
        self.model = attempt_load(weights=model_path, device=device)
        self.new_width = 320

    def __call__(self, frame):
        height, width = frame.shape[:2]
        new_height = int((((self.new_width/width)*height)//32)*32)

        img = cv2.resize(frame, (self.new_width, new_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).to(device)
        img = img.float()/255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        preds = self.model(img, augment=False)[0]
        preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, classes=None)
        items = []
        
        if preds[0] is not None and len(preds):
            for pred in preds[0]:
                # score = np.round(pred[4].cpu().detach().numpy(), 2)
                label = self.classes[int(pred[5])]
                xmin = int(pred[0] * width / self.new_width)
                ymin = int(pred[1] * height / new_height)
                xmax = int(pred[2] * width / self.new__width)
                ymax = int(pred[3] * height / new_height)

                item = {
                    'label': label,
                    'bbox' : [xmin, ymin, xmax, ymax]
                }

                items.append(item)

        return items