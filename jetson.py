import cv2
import argparse
import numpy as np
import nanocamera as nano

from ast import literal_eval

from assets import Assets
from model import ObjectDetection


def main(model_path, classes):
    assets = Assets()
    model = ObjectDetection(model_path)
    if classes == "all":
        class_arg = None
    else:
        class_arg = literal_eval(classes)

    camera = nano.Camera(flip=0, width=640, height=480, fps=30)
    while camera.isReady():
        try:
            frame = camera.read()
            objs = model(frame=frame, classes=class_arg)

            for obj in objs:
                score = obj['score']
                label = obj['label']
                xmin, ymin, xmax, ymax = obj['bbox']
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
                    org=(xmin,ymin),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX ,
                    fontScale=0.75,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA
                )

            cv2.imshow("video frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        except KeyboardInterrupt:
            break

    camera.release()
    del camera
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing models for inferencing.")
    parser.add_argument("-i", "--ip", default="127.0.0.1", type=str, help="IP address for stream.")
    parser.add_argument("-p", "--port", default=1935, type=str, help="Port for rtmp stream. Standard is 1935.")
    parser.add_argument("-a", "--application", default="live", type=str, help="Application name for server side.")
    parser.add_argument("-k", "--stream_key", default="stream", type=str, help="Stream key for security purposes.")
    parser.add_argument("-c", "--capture_index", default=0, type=int, help="Stream capture index. Most webcams are 0.")
    parser.add_argument("-m", "--model", default="weights/yolov5n.pt", type=str, help="Model to use. YOLO or local custom trained model.")
    parser.add_argument("--classes", type=str, help="Model to use. YOLO or local custom trained model.")
    args = parser.parse_args()
    
    main(
        args.model,
        args.classes
    )
