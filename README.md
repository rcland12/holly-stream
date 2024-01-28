# Holly Stream
This application will ingest your computers camera feed, apply an object detection task with bounding boxes, and send that feed via RTMP to an address of your choice. Watch the stream on your computer, another local machine, or an external web server.

# Requirements

* Docker and the compose plugin
* CUDA-enabled GPU
* At least 4GB of RAM
* Webcam, at least 720p recommended

# Prerequisites
There are a couple of recommended and required steps before you can run this application.

## (required) A YOLOv8 or YOLOv5 model
This application uses a service called Nvidia Trition Inference Server. It uses a object detection model format called TensorRT. THIS MODEL FORMAT IS MACHINE SPECIFIC. If I trained a YOLOv8 model on a Nvidia GeForce GTX 1060 GPU and convert it, it will not work on another GPU architecture. Therefore, you must convert your own model. If you want to use the default YOLOv8 model, you can simply follow the steps below. Otherwise, the next section will explain how to train a custom model with labels of your choice.

### How to convert a YOLOv8n (nano) model to TensorRT format

1. Set up a python environment.

2. Install YOLOv8 requirements:
    ```bash
    pip install ultralytics
    ```

3. Convert the model to ONNX format in a python script:

    ```python
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.export(
        format='onnx',   # export to ONNX
        half=True,       # FP16 quantization
        simplify=True    # ONNX model simplification
    )
    ```
    This model should save to `weights/yolov8n.onnx`.

4. Launch the Nvidia TensorRT container and convert the model:

    ```bash
    docker run -it --rm --gpus all -v ./weights:/models nvcr.io/nvidia/tensorrt:23.12-py3 \
    trtexec --onnx=/models/yolov8n.onnx --saveEngine=/models/model.plan --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
    ```
    This process should take a couple minutes.

5. Move the TensorRT model (`weights/model.plan`) to the triton directory:

    ```bash
    mv weights/model.plan triton/object_detection/1/
    ```

Triton should now be ready.

## (optional) Train a custom YOLOv5 model
This repository comes supplied with the default YOLOv5 medium model, trained on 80 [classes](#change-the-default-class-predictor). It is already in TensorRT format (`model.plan`) optimzed to run on Jetson Nano architecture. However, you can train your own custom model using the [YOLOv5 repo](https://github.com/ultralytics/yolov5). Follow these steps if you want to train a model from scratch. It is highly recommended to use a CUDA-enabled machine for training. If you already have a trained model in PyTorch format (`.pt`) skip to step 4.

1. Gather data. Collect images of the object(s) you want to detect. I created a script at `app/collect_data.py` to make this process easier.

2. Annotate your data in YOLO format. I highly recommend [Roboflow](https://roboflow.com/) for this and using a 70/30 split between your training and validation data.

3. Train your model. A CUDA-enabled machine is highly encouraged for this step. A training script will look something like this:
    ```bash
    python train.py \
    --weights yolov5s.pt \
    --cfg yolov5s.yaml \
    --data data.yaml \
    --epochs 1000 \
    --batch-size 16 \
    --optimizer AdamW \
    --device 0
    ```

4. After training, export your model as ONNX. Set the `--opset` to 15 for Jetson Nano compatability. If you want a pretrained YOLOv5 model, simply add the model name to `--weights` (e.g. `--weights yolov5s.pt`). The `--half` flg with convert model parameters that are float32 to float16, reducing model size and increasing inference speed.
    ```bash
    python export.py \
    --include onnx \
    --weights best.pt \
    --half \
    --opset 15 \
    --device 0
    ```

5. Copy your ONNX model to your Jetson Nano.

6. Convert your model to TensorRT. TensorRT model are specific to the machine you convert it on. If you are inferencing your model on a Jetson Nano you have create the TensorRT model on your Jetson Nano. Luckily, the software needed is built-in to the Jetson. If you did not use the `--half` flag in step 4, leave off the flags `--fp16` and after.
    ```bash
    /usr/src/tensorrt/bin/trtexec \
    --onnx=./path/to/model.onnx \
    --saveEngine=./path/to/save/model.plan \
    --explicitBatch \
    --fp16 \
    --inputIOFormats=fp16:chw \
    --outputIOFormats=fp16:chw
    ```

7. Move the model to the correct location, `holly-stream/triton/object_detection/1/model.plan`.

# Installation

Pull the docker images:
```bash
docker pull rcland12/detection-stream:linux-latest
docker pull rcland12/detection-stream:linux-triton-latest
```

# Deployment
You have the following options for serving this feed:

1. [Record a webcam feed from a Jetson Nano architecture.](#deploying-on-jetson-nano)
2. [Record a webcam feed from a Linux machine.](#deploying-on-a-linux-machine)

And the following options for serving this feed:

1. To the same device (localhost).
2. To another local device (LAN).
3. To a remote web server (WAN).

Lastly, you have two options for reading in this stream (client):

1. [Media player (VLC, Windows Media Player, etc.)](#watching-stream-through-streaming-software)
2. [Web page via Nginx/HLS.](#watching-stream-through-web-browser)

Pick any of the previous three options and follow the instructions below to deploy. If you are new to object detection I recommend you stick to the default model provided. Otherwise, you can supply your own YOLOv5 or YOLOv8 model.


## Deploying on Jetson Nano

If you are using a Jetson Nano you must have a camera attached. Begin by cloning the repository. Create an `.env` file to define parameters you wish to change. If you do not define a parameter it will default to a value given below, but you must define an empty `.env` file at minimum.

```bash
touch .env
```

Here is a list of all possible arguments:

```bash
OBJECT_DETECTION=True
TRITON_URL=grpc://localhost:8001
MODEL_NAME=yolov8n
MODEL_DIMS=(640, 640)
CLASSES=[0, 16]
CONFIDENCE_THRESHOLD=0.3
IOU_THRESHOLD=0.25

STREAM_IP=127.0.0.1
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=stream

CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
SANTA_HAT_PLUGIN=False
```

A few comments about the parameters:
- The `OBJECT_DETECTION` variable is a boolean to turn that tasks on/off. If you turn it off you simply have a live stream feed.
- You can use any custom trained YOLOv5 model for the `MODEL` parameter.
- The `CLASSES` variable takes in a list. If you wish to include all possible classes, remove it from the `.env` file. The possible classes for the default model are [listed below](#change-the-default-class-predictor).
- All arguments accept the data type present above. `STREAM_IP` takes a string, `STREAM_PORT` takes an integer, `STREAM_APPLICATION` takes a string, etc.

Next, if you intend to use the GPU for inference (highly recommended) you must enable the Nvidia runtime. Edit the file `/etc/docker/daemon.json` with the following changes:

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

1. [Watch the stream on streaming software such as VLC, Windows Media Player, OBS.](#watching-stream-through-streaming-software)
2. [Watch the stream through a browser.](#watching-stream-through-web-browser)

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
- All arguments accept the data type present above. `STREAM_IP` takes a string, `STREAM_PORT` takes an integer, `STREAM_APPLICATION` takes a string, etc.

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

1. [Watch the stream on streaming software such as VLC, Windows Media Player, OBS.](#watching-stream-through-streaming-software)
2. [Watch the stream through a browser.](#watching-stream-through-web-browser)

## Watching stream through streaming software
The client machine you are using must have Docker and the compose plugin. You will now launch a container that will pick up the feed and send it to your streaming software. If you already have Nginx running on port 1935 on your machine you will have to stop that service before you start this one.

```bash
docker compose up -d nginx-stream
```

Once the client software is running you can launch the streaming application from the server side (Jetson Nano or Linux). Run the following docker command if you are using a Jetson Nano:

```bash
docker-compose up -d jetson-stream
```

Or on a linux machine:
```bash
docker compose up -d linux-stream
```

I recommend appending the `-d` flag which will run the service in the background, but if you need to troubleshoot remove the `-d` flag. To close your running container, simply run `docker compose down`.

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
docker-compose up -d jetson-stream
```

Or on a linux machine:
```bash
docker compose up -d linux-stream
```

I recommend appending the `-d` flag which will run the service in the background, but if you need to troubleshoot remove the `-d` flag. To close your running container, simply run `docker compose down`.

Lastly, on the client side navigate to the web address `http://localhost/stream.html`. If you plan on serving this stream on a live web server, following the instructions in the next section.

## Streaming the feed to a web server
If you are streaming to a remote web server most of the steps will be the same. You will clone the repo on your web server. Before you launch the service you will have to make a few changes.

File `nginx/stream/stream.html`:
- Everywhere there is a `http://localhost`, replace it with your domain name: `https://website.com`.
- On line 20, replace the last part of the src, `stream.m3u8`, with your stream key: `<stream_key>.m3u8`. This prevents people on your same network from streaming a video to your website.

File `nginx/nginx-web/nginx.conf`:
- Whitelist your IP address by first finding your home IP address (you can do so [here](https://whatismyipaddress.com/)).
- Under the rtmp block, aroud lines 51-54 add another line with `allow publish <your_ipv4>;`. Don't forget the semicolon.
- Add your domain name on line 36. Replace `server_name localhost;` with `server_name website.com;`.Don't forget the semicolon.
- If for some reason you want to change the application name edit line 57 by replacing `application live` with `application <application_name>`. Then remember to also make that change in your `.env` file for the variable STREAM_APPLICATION.

Now you can launch this service on your web server:

```bash
docker compose run -d nginx-web
```

Once the client software is running you can launch the streaming application from the server side (Jetson Nano or Linux). Run the following docker command if you are using a Jetson Nano:

```bash
docker-compose up -d jetson-stream
```

Or on a linux machine:
```bash
docker compose up -d linux-stream
```

I recommend appending the `-d` flag which will run the service in the background, but if you need to troubleshoot remove the `-d` flag. To close your running container, simply run `docker compose down`.

Lastly, you will be able to access the stream at `https://website.com/stream.html`. You can of course make changes to this and create a different route for this stream, but this is the minimum requirements.


## Change the default class predictor
By default this application detects people and dogs (I made this for a home security system). To change or add classes for detection, add a CLASSES environmental variable to your `.env` file, if you don't already have it. Remove it to inference on all classes below. Otherwise, use a list to add classes you want to inference on. Such as `CLASSES=[0, 16, 17, 54, 67]`.

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
