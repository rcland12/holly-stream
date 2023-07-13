from ultralytics import YOLO
from supervision import Detections


def load_model_classes(model):
	model = YOLO(model)
	model.fuse()
	classes = model.model.names
	return model, classes


def plot_bboxes(results, frame, classes, box_annotator):
	xyxys = []
	confidences = []
	class_ids = []

	for result in results[0]:
		class_id = result.boxes.cls.cpu().numpy().astype(int)
		if class_id == 16:
			xyxys.append(result.boxes.xyxy.cpu().numpy())
			confidences.append(result.boxes.conf.cpu().numpy())
			class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

	detections = Detections(
		xyxy = results[0].boxes.xyxy.cpu().numpy(),
		confidence = results[0].boxes.conf.cpu().numpy(),
		class_id  = results[0].boxes.cls.cpu().numpy().astype(int)
	)

	labels = [
		f"{classes[class_id]} {confidence:0.2f}"
		for _, confidence, class_id, tracker_id
	   	in detections
	]

	frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

	return frame