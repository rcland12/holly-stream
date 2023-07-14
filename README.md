# Holly Stream
This application will ingest your computers webcam feed (using ffmpeg), apply an object detection task on the feed with bounding boxes, and send that feed via RTMP to an address of your choice.

## How to deploy
You can deploy with docker:
```bash
docker build -t holly-stream .
docker run -it --rm --net=host --device=/dev/video0:/dev/video0  holly-stream:latest
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

## Change the default class predictor
By default this application detects dogs. To change or add classes for detection, edit line 54 of `main.py`. For example is can be `classes=16` or `classes=[0, 14, 56]`. To include every class, change line 54 to `results = model(frame)`. The list of all possible classes are listed below:

| class_index  | class_name     |
|--------------|----------------|
| 0            | person         |
| 1            | bicycle        |
| 2            | car            |
| 3            | motorcycle     |
| 4            | airplane       |
| 5            | bus            |
| 6            | train          |
| 7            | truck          |
| 8            | boat           |
| 9            | traffic light  |
| 10           | fire hydrant   |
| 11           | stop sign      |
| 12           | parking meter  |
| 13           | bench          |
| 14           | bird           |
| 15           | cat            |
| 16           | dog            |
| 17           | horse          |
| 18           | sheep          |
| 19           | cow            |
| 20           | elephant       |
| 21           | bear           |
| 22           | zebra          |
| 23           | giraffe        |
| 24           | backpack       |
| 25           | umbrella       |
| 26           | handbag        |
| 27           | tie            |
| 28           | suitcase       |
| 29           | frisbee        |
| 30           | skis           |
| 31           | snowboard      |
| 32           | sports ball    |
| 33           | kite           |
| 34           | baseball bat   |
| 35           | baseball glove |
| 36           | skateboard     |
| 37           | surfboard      |
| 38           | tennis racket  |
| 39           | bottle         |
| 40           | wine glass     |
| 41           | cup            |
| 42           | fork           |
| 43           | knife          |
| 44           | spoon          |
| 45           | bowl           |
| 46           | banana         |
| 47           | apple          |
| 48           | sandwich       |
| 49           | orange         |
| 50           | brocolli       |
| 51           | carrot         |
| 52           | hot dog        |
| 53           | pizza          |
| 54           | donut          |
| 55           | cake           |
| 56           | chair          |
| 57           | couch          |
| 58           | potted plant   |
| 59           | bed            |
| 60           | dining table   |
| 61           | toilet         |
| 62           | tv             |
| 63           | laptop         |
| 64           | mouse          |
| 65           | remote         |
| 66           | keyboard       |
| 67           | cell phone     |
| 68           | microwave      |
| 69           | oven           |
| 70           | toaster        |
| 71           | sink           |
| 72           | refrigerator   |
| 73           | book           |
| 74           | clock          |
| 75           | vase           |
| 76           | scissors       |
| 77           | teddy bear     |
| 78           | hair drier     |
| 79           | toothbrush     |
