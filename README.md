# Holly Stream
This application will ingest your computers webcam feed (using ffmpeg), apply an object detection task on the feed with bounding boxes, and send that feed via RTMP to an address of your choice.

## How to deploy
You can deploy with docker:
```bash
docker build -t holly-stream .
docker run -it --rm holly-stream:latest
```

Or deploy in your own python environment:
```bash
pip install -r requirements.txt
python main.py
```

You can append flags to the python command via argparse if you do not want the default arguments. These are the default arguments:
```bash
python main.py \
--ip 127.0.0.1 \
--port 1935 \
--application live \
--stream_key stream \
--capture_index 0 \
--model yolov8n.pt
```
- Each argument can also be abbreviated (`--ip` can be just `-i-`, etc.).
- If you are hosting a live stream on a website you can send the feed to your web servers external IP address.
- The default port for FTMP video streams is 1935. If for some reason yours is different you can change it here.
- The application name is something that can be configured on server side. For instance, if you are using Nginx on your server side, the application name is defined in the nginx.conf.
- The stream key is a common parameter for streaming and allows for more security, ensuring others cannot tamper with your stream.
- The capture index is the index of your webcam or recording device. On linux this can be found at `/mnt/video0`.
- Lastly, the model allows you to use a PyTorch YOLO model or a custom trained model. This program uses YOLO in the backend, so only YOLO architecture will work here.

To change these parameters when using Docker, open the `Dockerfile` and make the changes. Then make sure to rebuild the container.
