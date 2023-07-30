import cv2

from assets import Assets
from nanocamera import Camera
from model import ObjectDetection
from utilities import EnvArgumentParser


def main(model_path, classes, width, height, fps):
    assets = Assets()
    model = ObjectDetection(model_path)

    camera = Camera(flip=0, width=width, height=height, fps=fps)
    while camera.isReady():
        try:
            frame = camera.read()
            objs = model(frame=frame, classes=classes)

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
        args.CAMERA_WIDTH,
        args.CAMERA_HEIGHT,
        args.CAMERA_FPS
    )
