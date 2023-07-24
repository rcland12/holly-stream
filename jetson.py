import cv2
import argparse
import subprocess
import nanocamera as nano


import io
import json
import torch
import argparse
import numpy as np

from utilities import (attempt_load, letterbox, scale_coords, non_max_suppression, normalize_boxes)

from PIL import Image

# from supervision import BoxAnnotator
# from supervision.draw.color import ColorPalette

from plotting import *


def main(ip_address, port, application, stream_key, capture_index, model):
	# set up object detection model, classes, and annotator
	# model, device = load_model(model)
	# box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=2)

    camera = nano.camera(flip=0, width=640, height=480, fps=30)
    print('CSI Camera Ready? -', camera.isReady())
    while camera.isReady():
        try:
            frame = camera.read()
            cv2.imshow("video frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break

    camera.release()
    del camera


    if torch.cuda.is_available():
        device = torch.device("cuda")
        fp16 = True
    else:
        device = torch.device("cpu")
        fp16 = False

    model = attempt_load(model, device=device)
    stride = max(int(model.stride.max()), 32)
    model.half() if fp16 else model.float()
    classes = model.names

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

		print(type(frame))
		# pil_img = Image.open(io.BytesIO(im_bytes)).convert("RGB")
		# open_cv_image = np.array(pil_img)
		# img0 = open_cv_image[:, :, ::-1].copy()
		# img = letterbox(img0, new_shape=IMG, stride=stride, auto=True)[0]
		# img = img.transpose((2, 0, 1))[::-1]
		# img = np.ascontiguousarray(img)
		# img = torch.from_numpy(img).to(device)
		# img = img.half() if fp16 else img.float()
		# img /= 255
		# if len(img.shape) == 3:
		# 	img = img[None,:,:,:]

		# results = model.predict(frame, classes=16, device=device, verbose=False)
		results = model(img, augment=True)[0]
		print(results)
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


# @app.route("/results", methods=["POST"])
# def predict():
#     if request.method != "POST":
#         return "Method was not POST"

#     models = paths_dict["models"]

#     dictionary = {}
#     if len(models) and request.files:

#         images = {}
#         for i in request.files:
#             img = request.files.get(i)
#             images[i] = img.read()

#         args = request.args
#         CONF = args.get("conf", default=0.6, type=float)
#         IOU = args.get("iou", default=0.1, type=float)
#         IMG = args.get("img", default=640, type=int)

#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#             fp16 = True
#         else:
#             device = torch.device("cpu")
#             fp16 = False

#         model = attempt_load(models, device=device)
#         stride = max(int(model.stride.max()), 32)
#         model.half() if fp16 else model.float()
#         classes = model.names

#         index = 0
#         for i in request.files:
#             im_bytes = images[i]
#             pil_img = Image.open(io.BytesIO(im_bytes)).convert("RGB")
#             open_cv_image = np.array(pil_img)
#             img0 = open_cv_image[:, :, ::-1].copy()
#             img = letterbox(img0, new_shape=IMG, stride=stride, auto=True)[0]
#             img = img.transpose((2, 0, 1))[::-1]
#             img = np.ascontiguousarray(img)
#             img = torch.from_numpy(img).to(device)
#             img = img.half() if fp16 else img.float()
#             img /= 255
#             if len(img.shape) == 3:
#                 img = img[None,:,:,:]

#             results = model(img, augment=True)[0]
#             preds = non_max_suppression(results, conf_thres=CONF, iou_thres=IOU)[0]
#             preds[:, :4] = scale_coords(img.shape[2:], preds[:, :4], img0.shape).round()

#             height, width = img0.shape[:2]
#             bboxes = [normalize_boxes(item[:4], width, height) for item in preds]
#             conf = [float(item[4]) for item in preds]
#             obj = [int(item[5]) for item in preds]
#             names = [classes[item] for item in obj]

#             if len(bboxes):
#                 for j in range(len(bboxes)):
#                     dictionary[index] = {"bboxes": bboxes[j],
#                                          "conf": conf[j],
#                                          "obj": obj[j],
#                                          "name": names[j],
#                                          "image_id": i
#                                         }
#                     index += 1

#         # with open("inference/output/results.json", "w") as f:
#         #     json.dump(dictionary, f)

#         return json.dumps(dictionary)

#     else:
#         return "Something went wrong. Either no model or image inputs"
