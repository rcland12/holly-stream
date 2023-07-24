import cv2
import argparse
import numpy as np

from assets import Assets
from model import ObjectDetection


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def main(model_path):
    assets = Assets()
    model = ObjectDetection(model_path)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret, frame = cap.read()
            if ret:
                objs = model(frame)

                for obj in objs:
                    label = obj['name']
                    score = obj['conf']
                    xmin, ymin, xmax, ymax = obj['bbox']
                    color = assets.colors[assets.classes.index(label)]
                    frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
                    frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)

            cv2.imshow("CSI Camera", frame)
            keyCode = cv2.waitKey(30)
            if keyCode == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing models for inferencing.")
    parser.add_argument("-i", "--ip", default="127.0.0.1", type=str, help="IP address for stream.")
    parser.add_argument("-p", "--port", default=1935, type=str, help="Port for rtmp stream. Standard is 1935.")
    parser.add_argument("-a", "--application", default="live", type=str, help="Application name for server side.")
    parser.add_argument("-k", "--stream_key", default="stream", type=str, help="Stream key for security purposes.")
    parser.add_argument("-c", "--capture_index", default=0, type=int, help="Stream capture index. Most webcams are 0.")
    parser.add_argument("-m", "--model", default="weights/yolov5n.pt", type=str, help="Model to use. YOLO or local custom trained model.")
    args = parser.parse_args()
    
    main(
        args.model
    )
