from ultralytics import YOLO
from supervision import Detections


#
# skips classes and labels
#
def load_model(model):
	model = YOLO(model)
	model.fuse()
	print(model.model.names)
	return model


def plot_bboxes(results, frame, box_annotator):
	xyxys = []

	for result in results[0]:
		xyxys.append(result.boxes.xyxy.cpu().numpy())

	detections = Detections(
		xyxy = results[0].boxes.xyxy.cpu().numpy()
	)

	frame = box_annotator.annotate(scene=frame, detections=detections, skip_label=True)
	return frame


#
# includes classes and labels
#
# def load_model(model):
# 	model = YOLO(model)
# 	model.fuse()
# 	classes = model.model.names
# 	return model, classes


# def plot_bboxes(results, frame, classes, box_annotator):
# 	xyxys = []
# 	confidences = []
# 	class_ids = []

# 	for result in results[0]:
# 		class_id = result.boxes.cls.cpu().numpy().astype(int)
# 		xyxys.append(result.boxes.xyxy.cpu().numpy())
# 		confidences.append(result.boxes.conf.cpu().numpy())
# 		class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

# 	detections = Detections(
# 		xyxy = results[0].boxes.xyxy.cpu().numpy(),
# 		confidence = results[0].boxes.conf.cpu().numpy()
# 		class_id  = results[0].boxes.cls.cpu().numpy().astype(int)
# 	)

# 	labels = [
# 		f"{classes[class_id]} {confidence:0.2f}"
# 		for _, mask, confidence, class_id, tracker_id
# 	  	in detections
# 	]

# 	frame = box_annotator.annotate(scene=frame, detections=detections)

# 	return frame
