import cv2
import torch
import subprocess

from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision import BoxAnnotator, Detections

from assets import EnvArgumentParser


# includes classes and labels
def load_model(model):
	device = "cuda" if torch.cuda.is_available() else "cpu"
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
		xyxys.append(result.boxes.xyxy.cpu().numpy())
		confidences.append(result.boxes.conf.cpu().numpy())
		class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

	detections = Detections(
		xyxy = results[0].boxes.xyxy.cpu().numpy(),
		confidence = results[0].boxes.conf.cpu().numpy()
		class_id  = results[0].boxes.cls.cpu().numpy().astype(int)
	)

	labels = [
		f"{classes[class_id]} {confidence:0.2f}"
		for _, mask, confidence, class_id, tracker_id
	  	in detections
	]

	frame = box_annotator.annotate(scene=frame, detections=detections)

	return frame


# skips classes and labels
# def load_model(model):
# 	model = YOLO(model)
# 	model.fuse()
# 	device = "cuda" if torch.cuda.is_available() else "cpu"
# 	return model, device


# def plot_bboxes(results, frame, box_annotator):
# 	xyxys = []

# 	for result in results[0]:
# 		xyxys.append(result.boxes.xyxy.cpu().numpy())

# 	detections = Detections(
# 		xyxy = results[0].boxes.xyxy.cpu().numpy()
# 	)

# 	frame = box_annotator.annotate(scene=frame, detections=detections, skip_label=True)
# 	return frame


def main(
		object_detection,
		model,
		classes,
		stream_ip,
		stream_port,
		stream_application,
		stream_key,
		camera_index
	):

	if object_detection:
		model, device = load_model(model)
		box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=2)

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

		if object_detection:
			results = model.predict(frame, classes=classes, device=device, verbose=False)
			frame = plot_bboxes(results, frame, box_annotator)

		p.stdin.write(frame.tobytes())


if __name__ == "__main__":
    parser = EnvArgumentParser()
    parser.add_arg("OBJECT_DETECTION", default=True, type=bool)
    parser.add_arg("MODEL", default="weights/yolov8n.pt", type=str)
    parser.add_arg("CLASSES", default=None, type=list)
    parser.add_arg("STREAM_IP", default="127.0.0.1", type=str)
    parser.add_arg("STREAM_PORT", default=1935, type=int)
    parser.add_arg("STREAM_APPLICATION", default="live", type=str)
    parser.add_arg("STREAM_KEY", default="stream", type=str)
    parser.add_arg("CAMERA_INDEX", default=0, type=int)
    args = parser.parse_args()

    main(
	    args.OBJECT_DETECTION,
	    args.MODEL,
        args.CLASSES,
	    args.STREAM_IP,
	    args.STREAM_PORT,
	    args.STREAM_APPLICATION,
	    args.STREAM_KEY,
	    args.CAMERA_INDEX
    )







# import cv2
# import torch
# import subprocess

# from ultralytics import YOLO
# from supervision.draw.color import ColorPalette
# from supervision import BoxAnnotator, Detections

# from assets import Assets
# from model import ObjectDetection
# from utilities import EnvArgumentParser



# def main(
# 		object_detection,
# 		model_name,
# 		classes,
# 		confidence_threshold,
# 		iou_threshold,
# 		stream_ip,
# 		stream_port,
# 		stream_application,
# 		stream_key,
# 		camera_index
# 	):

# 	rtmp_url = "rtmp://{}:{}/{}/{}".format(
# 		stream_ip,
# 		stream_port,
# 		stream_application,
# 		stream_key
# 	)

# 	capture = cv2.VideoCapture(camera_index)
# 	camera_fps = int(capture.get(cv2.CAP_PROP_FPS))
# 	camera_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# 	camera_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 	command = [
# 		'ffmpeg',
# 		'-y',
# 		'-f', 'rawvideo',
# 		'-vcodec', 'rawvideo',
# 		'-pix_fmt', 'bgr24',
# 		'-s', "{}x{}".format(camera_width, camera_height),
# 		'-r', str(camera_fps),
# 		'-i', '-',
# 		'-c:v', 'libx264',
# 		'-pix_fmt', 'yuv420p',
# 		'-preset', 'ultrafast', 
# 		'-f', 'flv',
# 		rtmp_url
# 	]

# 	p = subprocess.Popen(command, stdin=subprocess.PIPE)

# 	if object_detection:
# 		assets = Assets()
# 		model = ObjectDetection(
# 			model_name=model_name,
# 			all_classes=assets.classes,
# 			classes=classes,
# 			camera_width=camera_width,
# 			camera_height=camera_height,
# 			confidence_threshold=confidence_threshold,
# 			iou_threshold=iou_threshold,
# 			triton_url="http://localhost:8000"
# 		)

# 		while capture.isOpened():
# 			ret, frame = capture.read()
# 			if not ret:
# 				print("Frame read failed")
# 				break

# 			bboxes, confs, indexes = model(frame)

# 			for i in range(len(bboxes)):
# 				xmin, ymin, xmax, ymax = bboxes[i]
# 				color = assets.colors[indexes[i]]
# 				frame = cv2.rectangle(
# 					img=frame,
# 					pt1=(xmin, ymin),
# 					pt2=(xmax, ymax),
# 					color=color,
# 					thickness=2
# 				)
# 				frame = cv2.putText(
# 					img=frame,
# 					text=f'{assets.classes[indexes[i]]} ({str(confs[i])})',
# 					org=(xmin, ymin),
# 					fontFace=cv2.FONT_HERSHEY_PLAIN ,
# 					fontScale=0.75,
# 					color=color,
# 					thickness=1,
# 					lineType=cv2.LINE_AA
# 				)

# 			p.stdin.write(frame.tobytes())

# 	else:
# 		while capture.isOpened():
# 			ret, frame = capture.read()
# 			if not ret:
# 				print("Frame read failed")
# 				break

# 			p.stdin.write(frame.tobytes())

# if __name__ == "__main__":
# 	parser = EnvArgumentParser()
# 	parser.add_arg("OBJECT_DETECTION", default=True, type=bool)
# 	parser.add_arg("MODEL", default="weights/yolov8n.pt", type=str)
# 	parser.add_arg("CLASSES", default=None, type=list)
# 	parser.add_arg("CONFIDENCE_THRESHOLD", default=0.3, type=float)
# 	parser.add_arg("IOU_THRESHOLD", default=0.45, type=float)
# 	parser.add_arg("STREAM_IP", default="127.0.0.1", type=str)
# 	parser.add_arg("STREAM_PORT", default=1935, type=int)
# 	parser.add_arg("STREAM_APPLICATION", default="live", type=str)
# 	parser.add_arg("STREAM_KEY", default="stream", type=str)
# 	parser.add_arg("CAMERA_INDEX", default=0, type=int)
# 	args = parser.parse_args()

# 	main(
# 	    args.OBJECT_DETECTION,
# 	    args.MODEL,
#         args.CLASSES,
# 		args.CONFIDENCE_THRESHOLD,
# 		args.IOU_THRESHOLD,
# 	    args.STREAM_IP,
# 	    args.STREAM_PORT,
# 	    args.STREAM_APPLICATION,
# 	    args.STREAM_KEY,
# 	    args.CAMERA_INDEX
#     )
