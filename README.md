# Holly Stream
This application will ingest your computers webcam feed (using ffmpeg), apply an object detection task on the feed with bounding boxes, and send that feed via RTMP to an address of your choice. You have the following options for recording and applying a custom object detection model:

1. Record a webcam feed from a Linux machine.
2. Record a webcam feed from a Jetson Nano architecture.

And the following options for serving this feed:

1. To the same device (localhost).
2. To another local device (LAN).
3. To a remote web server (WAN).

Lastly, you have two options for reading in this stream (client):

1. Media player (VLC, Windows Media Player, etc.)
2. Web page via Nginx/HLS.

Pick any of the previous three options and follow the instructions below to deploy. If you are new to object detection I recommend you stick to the default model provided. Otherwise, you can supply your own YOLOv5 or YOLOv8 model.

# Deployment

## Deploying on Jetson Nano

If you are using a Jetson Nano you must have a camera attached. Begin by cloning the repository. Create an `.env` file to define parameters you wish to change. If you do not define a parameter it will default to a value given below, but you must define an empty `.env` file at minimum.

```bash
touch .env
```

Here is a list of all possible arguments:

```bash
OBJECT_DETECTION=True
MODEL=weights/yolov5n.pt
CLASSES=[0, 16]

STREAM_IP=127.0.0.1
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=stream

CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30
```

A few comments about the parameters:
- The `OBJECT_DETECTION` variable is a boolean to turn that tasks on/off. If you turn it off you simply have a live stream feed.
- The `CLASSES` variable takes in a list. If you wish to include all possible classes, remove it from the `.env` file.
- All classes accept the data type present above. `STREAM_IP` takes a string, `STREAM_PORT` takes an integer, `STREAM_APPLICATION` takes a string, etc.

If you have a GPU on your linux machine (recommended) then append the following block to the `docker-compose.yml` file under the service `linux-service`:
```yaml
deploy:
    resources:
    reservations:
        devices:
        - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Next, to launch the application run:
```bash
docker compose run linux-service
```

## Deploying on a Linux machine

If you are using a 

Define an environmental variable for the path to this repo:
```bash
export STREAM_PATH=<path_to_repo>

# example
export STREAM_PATH=/home/russ/holly-stream
```

## How to deploy on a Linux machine
You can deploy with docker:
```bash
docker build -t holly-stream .
docker run -it --rm --net=host --device=/dev/video0:/dev/video0  holly-stream:latest
```

Or deploy in your own python environment:
```bash
# install prerequisites
sudo apt update
sudo apt install ffmpeg
pip install -r requirements.txt

# run app
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
- Each argument can also be abbreviated (`--ip` can be just `-i`, etc.).
- If you are hosting a live stream on a website you can send the feed to your web servers external IP address.
- The default port for FTMP video streams is 1935. If for some reason yours is different you can change it here.
- The application name is something that can be configured on server side. For instance, if you are using Nginx on your server side, the application name is defined in the nginx.conf.
- The stream key is a common parameter for streaming and allows for more security, ensuring others cannot tamper with your stream.
- The capture index is the index of your webcam or recording device. On linux this can be found at `/mnt/video0`.
- Lastly, the model allows you to use a PyTorch YOLO model or a custom trained model. This program uses YOLO in the backend, so only YOLO architecture will work here.

To change these parameters when using Docker, open the `Dockerfile` and make the changes. Then make sure to rebuild the container.

## How to deploy on Nvidia Jetson Nano SDK
The default operating system on the Jetson Nane is Ubuntu 18.04 with Python version 3.6 and JetPack version 4.6.1.

Installing docker-compose on Jetson Nano:
```bash
pip3 install --upgrade pip
pip3 install docker-compose==1.27.4
```

Check if it was installed correctly:
```bash
docker-compose version
```

If you plan on building this container you have to define the default Docker runtime by editing the file `/etc/docker/daemon.json` with the changes below. If you pull a pre-built container then you can define the runtime in your `docker run` command with the flag `--runtime=nvidia`.
```bash
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

To deploy with Docker run the following script:
```bash
./run.sh
```

To build the container from source follow these commands below. This build takes roughly two hours to complete, so the previous option is advisable unless you made changes to the Dockerfile.
```bash
docker build -f Dockerfile.jetson -t detection-stream:jetson-latest
docker run --rm \
--interactive \
--tty \
--net=host \
--env DISPLAY=$DISPLAY \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--volume /tmp/argus_socket:/tmp/argus_socket \
--volume ${STREAM_PATH}/weights:/root/app/weights \
detection-stream:latest
```

To deploy a YOLOv5 model on the Nvidia Jetson Nano, I have found it easiest to use the following versions:
- Python 3.6.9 (default)
- OpenCV 4.5.1
- Pytorch 1.8.0/Torchvision 0.9.0 ([installation instructions](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048))

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
