import cv2
import subprocess

from assets import Assets
from nanocamera import Camera
from model import ObjectDetection
from utilities import EnvArgumentParser


def main(
		model_path,
		classes,
		stream_ip,
		stream_port,
		stream_application,
		stream_key,
		camera_index,
		camera_width,
		camera_height,
		camera_fps
	):
	
	assets = Assets()
	model = ObjectDetection(model_path)

	rtmp_url = "rtmp://{}:{}/{}/{}".format(
		stream_ip,
		stream_port,
		stream_application,
		stream_key
	)

	camera = Camera(
		device_id=camera_index,
		flip=0,
		width=camera_width,
		height=camera_height,
		fps=camera_fps
	)

	command = [
		'ffmpeg',
		'-y',
		'-f', 'rawvideo',
		'-vcodec', 'rawvideo',
		'-pix_fmt', 'bgr24',
		'-s', "{}x{}".format(camera_width, camera_height),
		'-r', str(camera_fps),
		'-i', '-',
		'-c:v', 'libx264',
		'-pix_fmt', 'yuv420p',
		'-preset', 'ultrafast', 
		'-f', 'flv',
		rtmp_url
	]

	p = subprocess.Popen(command, stdin=subprocess.PIPE)

	while camera.isReady():
		frame = camera.read()
		results = model(frame=frame, classes=classes)

		for result in results:
			score = result['score']
			label = result['label']
			xmin, ymin, xmax, ymax = result['bbox']
			color = assets.colors[assets.classes.index(label)]
			frame = cv2.rectangle(
				img=frame,
				pt1=(xmin, ymin),
				pt2=(xmax, ymax),
				color=color,
				thickness=2
			)
			frame = cv2.putText(
				img=frame,
				text=f'{label} ({str(score)})',
				org=(xmin, ymin),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX ,
				fontScale=0.75,
				color=color,
				thickness=1,
				lineType=cv2.LINE_AA
			)

		p.stdin.write(frame.tobytes())

	camera.release()
	del camera


if __name__ == "__main__":
    parser = EnvArgumentParser()
    parser.add_arg("MODEL", default="weights/yolov5n.pt", type=str)
    parser.add_arg("CLASSES", default=None, type=list)
    parser.add_arg("STREAM_IP", default="127.0.0.1", type=str)
    parser.add_arg("STREAM_PORT", default=1935, type=int)
    parser.add_arg("STREAM_APPLICATION", default="live", type=str)
    parser.add_arg("STREAM_KEY", default="stream", type=str)
    parser.add_arg("CAMERA_INDEX", default=0, type=int)
    parser.add_arg("CAMERA_WIDTH", default=640, type=int)
    parser.add_arg("CAMERA_HEIGHT", default=480, type=int)
    parser.add_arg("CAMERA_FPS", default=30, type=int)
    args = parser.parse_args()

    main(
        args.MODEL,
        args.CLASSES,
        args.STREAM_IP,
        args.STREAM_PORT,
        args.STREAM_APPLICATION,
        args.STREAM_KEY,
        args.CAMERA_INDEX,
        args.CAMERA_WIDTH,
        args.CAMERA_HEIGHT,
        args.CAMERA_FPS
    )
