import cv2
import torch
import subprocess

from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision import BoxAnnotator, Detections

from utilities import EnvArgumentParser


# skips classes and labels
def load_model(model):
	model = YOLO(model)
	model.fuse()
	device = "cuda" if torch.cuda.is_available() else "cpu"
	return model, device


def plot_bboxes(results, frame, box_annotator):
	xyxys = []

	for result in results[0]:
		xyxys.append(result.boxes.xyxy.cpu().numpy())

	detections = Detections(
		xyxy = results[0].boxes.xyxy.cpu().numpy()
	)

	frame = box_annotator.annotate(scene=frame, detections=detections, skip_label=True)
	return frame


# includes classes and labels
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


def main(
		model,
		classes,
		stream_ip,
		stream_port,
		stream_application,
		stream_key,
		camera_index
	):
	# set up object detection model, classes, and annotator
	model, device = load_model(model)
	box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=2)

	# set up stream parameters
	rtmp_url = "rtmp://{}:{}/{}/{}".format(
		stream_ip,
		stream_port,
		stream_application,
		stream_key
	)

	cap = cv2.VideoCapture(camera_index)

	fps = int(cap.get(cv2.CAP_PROP_FPS))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	command = [
		'ffmpeg',
		'-y',
		'-f', 'rawvideo',
		'-vcodec', 'rawvideo',
		'-pix_fmt', 'bgr24',
		'-s', "{}x{}".format(width, height),
		'-r', str(fps),
		'-i', '-',
		'-c:v', 'libx264',
		'-pix_fmt', 'yuv420p',
		'-preset', 'ultrafast', 
		'-f', 'flv',
		rtmp_url
	]

	p = subprocess.Popen(command, stdin=subprocess.PIPE)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("Frame read failed")
			break

		results = model.predict(frame, classes=classes, device=device, verbose=False)
		frame = plot_bboxes(results, frame, box_annotator)

		p.stdin.write(frame.tobytes())


if __name__ == "__main__":
    parser = EnvArgumentParser()
    parser.add_arg("STREAM_IP", default="127.0.0.1", type="str")
    parser.add_arg("STREAM_PORT", default=1935, type="int")
    parser.add_arg("STREAM_APPLICATION", default="live", type="str")
    parser.add_arg("STREAM_KEY", default="stream", type="str")
    parser.add_arg("CAMERA_INDEX", default=0, type="int")
    parser.add_arg("MODEL", default="weights/yolov5n.pt", type="str")
    parser.add_arg("CLASSES", default=None, type="list")
    args = parser.parse_args()

    main(
	    args["MODEL"],
        args["CLASSES"],
	    args["STREAM_IP"],
	    args["STREAM_PORT"],
	    args["STREAM_APPLICATION"],
	    args["STREAM_KEY"],
	    args["CAMERA_INDEX"]
    )