# Holly Stream
This application will ingest your computers webcam feed (using ffmpeg), apply an object detection task on the feed with bounding boxes, and send that feed via RTMP to an address of your choice. You have the following options for recording and applying a custom object detection model:

1. [Record a webcam feed from a Linux machine.]()
2. [Record a webcam feed from a Jetson Nano architecture.]()

And the following options for serving this feed:

1. To the same device (localhost).
2. To another local device (LAN).
3. To a remote web server (WAN).

Lastly, you have two options for reading in this stream (client):

1. [Media player (VLC, Windows Media Player, etc.)]()
2. [Web page via Nginx/HLS.]()

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
- You can use any custom trained YOLOv5 model for the `MODEL` parameter.
- The `CLASSES` variable takes in a list. If you wish to include all possible classes, remove it from the `.env` file. The possible classes for the default model are [listed below]().
- All classes accept the data type present above. `STREAM_IP` takes a string, `STREAM_PORT` takes an integer, `STREAM_APPLICATION` takes a string, etc.

To launch this application you need the docker-compose plugin installed. Install using the following command:

```bash
pip3 install --upgrade pip
pip3 install docker-compose==1.27.4
```

Check if it was installed correctly:

```bash
docker-compose version
```

Next, if you intend to use the GPU for inference you must enable the Nvidia runtime. Edit the file `/etc/docker/daemon.json` with the following changes:

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

The application is ready to launch, so pick the method for receiving the stream:

1. If you are watching the stream on the same device set the `STREAM_IP` to `127.0.0.1`, or whatever localhost address you want.
2. If you are streaming the feed to another device on your local network, set the `STREAM_IP` to the IPV4 address of that device. To find the IP address of a device on linux machine run `ip a` and look for it under `wl01` or something similar. On a windows machine open Windows Powershell and run `ipconfig` and look for `IP address`.
3. If you are streaming the feed to another server set the `STREAM_IP` to the public IP address for that server. Also make sure you expose port 1935 on the firewall and router if necessary for the server your sending the stream to.

Lastly, to receive the stream on the device you picked above you have two options:

1. [Watch the stream on streaming software such as VLC, Windows Media Player, OBS.](https://google.com)
2. [Watch the stream through a browser.](https://google.com)

## Deploying on a Linux machine

You must have a webcam attached to your Linux machine. Begin by cloning the repository. Create an `.env` file to define parameters you wish to change. If you do not define a parameter it will default to a value given below, but you must define an empty `.env` file at minimum.

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
- You can use any custom trained YOLOv5 model for the `MODEL` parameter.
- The `CLASSES` variable takes in a list. If you wish to include all possible classes, remove it from the `.env` file. The possible classes for the default model are [listed below]().
- All classes accept the data type present above. `STREAM_IP` takes a string, `STREAM_PORT` takes an integer, `STREAM_APPLICATION` takes a string, etc.

If you have an CUDA enabled Nvidia GPU it is highly recommended you add the following block to the `docker-compose.yml` file under the service `linux-stream`.This will allow you to inference on your GPU, which is highly optimal for matrix multiplications.
```bash
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

The application is ready to launch, so pick the method for receiving the stream:

1. If you are watching the stream on the same device set the `STREAM_IP` to `127.0.0.1`, or whatever localhost address you want.
2. If you are streaming the feed to another device on your local network, set the `STREAM_IP` to the IPV4 address of that device. To find the IP address of a device on linux machine run `ip a` and look for it under `wl01` or something similar. On a windows machine open Windows Powershell and run `ipconfig` and look for `IP address`.
3. If you are streaming the feed to another server set the `STREAM_IP` to the public IP address for that server. Also make sure you expose port 1935 on the firewall and router if necessary for the server your sending the stream to.

Lastly, to receive the stream on the device you picked above you have two options:

1. [Watch the stream on streaming software such as VLC, Windows Media Player, OBS.](https://google.com)
2. [Watch the stream through a browser.](https://google.com)

## Watching stream through streaming software.
The client machine you are using must have Docker and the compose plugin. You will now launch a container that will pick up the feed and send it to your streaming software. If you already have Nginx running on port 1935 on your machine you will have to stop that service before you start this one.

```bash
docker compose up -d nginx-stream
```

Once the client software is running you can launch the streaming application from the server side (Jetson Nano or Linux). Run the following docker command if you are using a Jetson Nano:

```bash
docker-compose up jetson-stream
```

Or on a linux machine:
```bash
docker compose up linux-stream
```

Lastly, on the client side you can open up your streaming software and find where you can watch a network stream or URL stream, then use the address you set up in the parameters:
```bash
rtmp://<STREAM_IP>:<STREAM_PORT>/<STREAM_APPLICATION>/<STREAM_KEY>

# example
rtmp://127.0.0.1:1935/live/stream
```

## Watching stream through web browser
The client machine you are using must have Docker and the compose plugin. You will now launch a container that will start a web server on localhost port 80. If you already have Nginx running on port 1935 on your machine you will have to stop that service before you start this one.

```bash
docker compose up -d nginx-web
```

Once the client software is running you can launch the streaming application from the server side (Jetson Nano or Linux). Run the following docker command if you are using a Jetson Nano:

```bash
docker-compose up jetson-stream
```

Or on a linux machine:
```bash
docker compose up linux-stream
```

Lastly, on the client side navigate to the web address `http://localhost/stream.html`. If you plan on serving this stream on a live web server, following the instructions in the next section.

## Streaming the feed to a web server



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
