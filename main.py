import argparse
import cv2 as cv
import subprocess

from supervision import BoxAnnotator
from supervision.draw.color import ColorPalette

from predict import *


def main(ip_address, port, application, stream_key, capture_index, model):
	# set up object detection model, classes, and annotator
	model = load_model(model)
	box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=2)

	# set up stream parameters
	rtmp_url = "rtmp://{}:{}/{}/{}".format(
		ip_address,
		port,
		application,
		stream_key
	)

	cap = cv.VideoCapture(capture_index)

	fps = int(cap.get(cv.CAP_PROP_FPS))
	width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

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

		results = model.predict(frame, classes=16)
		frame = plot_bboxes(results, frame, box_annotator)

		p.stdin.write(frame.tobytes())


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Flask API exposing models for inferencing.")
	parser.add_argument("-i", "--ip", default="127.0.0.1", type=str, help="IP address for stream.")
	parser.add_argument("-p", "--port", default=1935, type=str, help="Port for rtmp stream. Standard is 1935.")
	parser.add_argument("-a", "--application", default="live", type=str, help="Application name for server side.")
	parser.add_argument("-k", "--stream_key", default="stream", type=str, help="Stream key for security purposes.")
	parser.add_argument("-c", "--capture_index", default=0, type=int, help="Stream capture index. Most webcams are 0.")
	parser.add_argument("-m", "--model", default="yolov8n.pt", type=str, help="Model to use. YOLO or local custom trained model.")
	args = parser.parse_args()

	main(
		args.ip,
		args.port,
		args.application,
		args.stream_key,
		args.capture_index,
		args.model
	)
