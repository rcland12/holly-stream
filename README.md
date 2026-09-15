# Holly-Stream

<img src="./app/images/logo.png" alt="Holly-Stream Logo" style="width: auto;">

A containerized live streaming application for Raspberry Pi featuring real-time object detection, motion detection, and multi-protocol streaming capabilities. Built with YOLOv8, Triton Inference Server, and FFmpeg for high-performance video processing.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deployment Options](#deployment-options)
- [Advanced Configuration](#advanced-configuration)
- [Custom Model Training](#custom-model-training)
- [Multi-Camera Setup](#multi-camera-setup)
- [Troubleshooting](#troubleshooting)

## Overview

Holly-Stream transforms your Raspberry Pi into a powerful streaming platform capable of:

- **Real-time Object Detection**: Using YOLOv8 models with Nvidia Triton Inference Server for optimized performance
- **Motion Detection**: Automated video recording when motion is detected, with remote upload capabilities
- **Live Streaming**: Direct camera feed streaming without processing overhead
- **Multi-Protocol Support**: Stream via RTMP to media players or HLS for web browsers
- **Flexible Deployment**: Local network, remote web server, or localhost streaming options

## Features

### Core Capabilities

- **Object Detection Pipeline**
  - YOLOv8 model inference with ONNX format
  - 80 COCO classes supported by default (customizable)
  - Configurable confidence and IOU thresholds
  - Real-time bounding box annotation
  - Selective class filtering

- **Motion Detection System**
  - SSIM-based motion detection algorithm
  - Automatic video recording on motion trigger
  - Remote server upload via SCP
  - Configurable sensitivity and recording duration

- **Camera Support**
  - Raspberry Pi Camera Module (via libcamera/rpicam-vid)
  - Legacy Raspberry Pi cameras (via raspivid)
  - USB webcams (via V4L2)
  - ArduCam modules (IMX519 tested)
  - Audio capture from USB or built-in sources

- **Streaming Protocols**
  - RTMP streaming for media players (VLC, Windows Media Player)
  - HLS streaming for web browsers
  - Configurable quality presets (Ultra, High, Medium, Low)
  - Hardware-accelerated H.264 encoding when available

- **Additional Features**
  - Santa Hat Plugin (novelty overlay for detected objects)
  - Multi-camera orchestration and remote management
  - Data collection utility for custom model training
  - Automatic failover and restart capabilities

## Architecture

### System Overview

```mermaid
flowchart TB
    subgraph RaspberryPi["Raspberry Pi Host"]
        Camera["Camera Device<br/>/dev/video* or libcamera"]

        subgraph DockerEnv["Docker Environment"]
            subgraph AppContainer["App Container"]
                Picamera["Picamera2<br/>Frame Capture"]
                Detector["Detection Logic"]
                Annotator["Frame Annotation"]
                FFmpeg["FFmpeg Encoder<br/>H.264/RTMP"]
            end

            subgraph TritonContainer["Triton Container"]
                Preprocess["Preprocess Model<br/>Letterbox + Normalize"]
                YOLOModel["YOLOv8 ONNX<br/>Object Detection"]
                Postprocess["Postprocess Model<br/>NMS + Filtering"]
            end

            subgraph NginxContainer["Nginx Container"]
                RTMPServer["RTMP Server<br/>Port 1935"]
                HLSConverter["HLS Converter<br/>3s segments"]
                WebServer["HTTP Server<br/>Port 8080"]
            end
        end
    end

    subgraph Clients["Client Devices"]
        MediaPlayer["Media Players<br/>VLC, WMP"]
        WebBrowser["Web Browsers<br/>HLS.js Player"]
    end

    Camera -->|Video Feed| Picamera
    Picamera -->|Raw Frames| Detector
    Detector <-->|HTTP/gRPC| Preprocess
    Preprocess --> YOLOModel
    YOLOModel --> Postprocess
    Postprocess -->|Detections| Detector
    Detector -->|Annotated Frames| Annotator
    Annotator -->|RGB Frames| FFmpeg
    FFmpeg -->|RTMP Stream| RTMPServer
    RTMPServer -->|HLS Segments| HLSConverter
    HLSConverter --> WebServer
    RTMPServer -.->|Direct RTMP| MediaPlayer
    WebServer -->|HTTP/HLS| WebBrowser

    style AppContainer fill:#e1f5ff
    style TritonContainer fill:#fff4e1
    style NginxContainer fill:#e8f5e9
    style RaspberryPi fill:#f5f5f5
```

### Data Flow

**Object Detection Mode:**
1. Camera captures frame at configured FPS (e.g., 30 FPS)
2. Every 10th frame sent to Triton Inference Server
3. Triton pipeline: Preprocess → Detection → Postprocess
4. Detection results returned to application
5. Frame annotated with bounding boxes and labels
6. All frames encoded with FFmpeg and streamed via RTMP
7. Nginx receives RTMP stream and converts to HLS
8. Clients connect via RTMP or HTTP/HLS

**Motion Detection Mode:**
1. Camera captures frames at 1 FPS
2. SSIM algorithm compares consecutive frames
3. When similarity drops below threshold, motion detected
4. Records H.264 video for configured duration
5. Uploads video to remote server via SCP
6. Returns to monitoring state

**Direct Streaming Mode:**
1. Camera feed piped directly to FFmpeg
2. H.264 encoding with minimal latency
3. RTMP output to nginx server
4. No processing overhead

## System Requirements

### Hardware

- **Raspberry Pi 4 or 5** (4GB+ RAM recommended)
- **Camera**: 720p minimum, 1080p recommended
  - Raspberry Pi Camera Module v2/v3
  - ArduCam (IMX519 tested)
  - USB webcam
- **Storage**: 8GB+ microSD card
- **Network**: Ethernet or WiFi connection

### Software

- **Operating System**: Raspberry Pi OS (64-bit, Bookworm or later)
- **Docker**: Version 20.10 or later with compose plugin
- **FFmpeg**: Version 4.3 or later (installed via setup script)
- **Camera Drivers**:
  - libcamera for modern Pi cameras
  - ArduCam drivers if using ArduCam modules ([installation guide](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/Quick-Start-Guide/))

### Optional Requirements

- **For Object Detection**: Nvidia Triton-compatible model repository
- **For Motion Detection**: Remote server with SSH access for video uploads
- **For Custom Models**: CUDA-enabled machine for training (not required for inference)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/holly-stream.git
cd holly-stream
```

### 2. Run Setup Script

Install dependencies and configure the environment:

```bash
chmod +x ./app/setup.sh
./app/setup.sh
```

This script will install FFmpeg, camera utilities, and Python dependencies.

### 3. Pull Docker Images

Pull the pre-built container images:

```bash
# For object detection
docker pull rcland12/detection-stream:raspbian-triton-latest

# For nginx streaming server
docker pull rcland12/detection-stream:nginx-latest

# For main application
docker pull rcland12/detection-stream:raspbian-latest
```

### 4. Make Scripts Executable

```bash
chmod +x ./run.sh ./stop.sh
```

### 5. Camera Setup

**For Raspberry Pi Camera Module:**
- Ensure camera is properly connected to CSI port
- Enable camera in `raspi-config` if using legacy camera
- Verify with: `libcamera-hello` (modern) or `raspistill -t 0` (legacy)

**For ArduCam:**
- Install drivers: [ArduCam Quick Start Guide](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/Quick-Start-Guide/)
- Verify camera is detected: `v4l2-ctl --list-devices`

**For USB Webcam:**
- Connect camera to USB port
- Verify device: `ls /dev/video*`
- Note device number (usually `/dev/video0`)

## Configuration

### Environment Variables

Create a `.env` file in the project root directory. This file controls all operational parameters.

**Required Variables:**

```bash
# Operation Mode (required)
OBJECT_DETECTION=True  # True/False - enables/disables object detection
```

**Camera Configuration:**

```bash
# Camera Settings
CAMERA_WIDTH=1280           # Resolution width in pixels
CAMERA_HEIGHT=720           # Resolution height in pixels
CAMERA_FPS=30              # Frames per second
CAMERA_AUDIO=True          # Include audio in stream (True/False)
CAMERA_BACKEND=auto        # auto, libcamera, legacy, or usb
CAMERA_INDEX=0             # USB camera device index (/dev/videoX)
```

**Streaming Configuration:**

```bash
# RTMP Stream Settings
STREAM_IP=127.0.0.1        # Target server IP address
STREAM_PORT=1935           # RTMP port (default: 1935)
STREAM_APPLICATION=live    # RTMP application name
STREAM_KEY=stream          # RTMP stream key
STREAM_USER=username       # Username for SCP uploads (motion detection)
```

**Object Detection Settings:**

```bash
# Model Configuration
TRITON_URL=grpc://127.0.0.1:8001      # Triton endpoint (grpc:// or http://)
MODEL_NAME=yolo11                     # Ensemble name in triton/repository
MODEL_REPOSITORY=/models              # Model repository path inside the Triton container
CONFIDENCE_THRESHOLD=0.3              # Minimum score to draw (the model already drops < 0.25)
CLASSES="[0, 16]"                     # Class indexes to draw, "[]" for all (see labels.txt)

# AWS S3 Model Repository (optional)
AWS_ACCESS_KEY_ID=your_key_id
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

**Motion Detection Settings:**

```bash
# Motion Detection
MOTION_DETECTION=False     # Enable motion detection (True/False)
MOTION_THRESHOLD=0.75      # SSIM threshold (0.0-1.0, lower = more sensitive)
VIDEO_LENGTH=30            # Recording duration in seconds
VIDEOS_FILE_PATH=/app/videos  # Remote server destination path
```

**Special Features:**

```bash
# Santa Hat Plugin
SANTA_HAT_PLUGIN=False     # Overlay santa hat on highest confidence detection
```

### Configuration Examples

**Example 1: Local Object Detection**
```bash
OBJECT_DETECTION=True
STREAM_IP=127.0.0.1
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=stream
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30
CONFIDENCE_THRESHOLD=0.5
CLASSES="[0, 16]"  # Detect only persons and dogs
```

**Example 2: Motion Detection with Remote Upload**
```bash
OBJECT_DETECTION=False
MOTION_DETECTION=True
STREAM_USER=pi
STREAM_IP=192.168.1.100
MOTION_THRESHOLD=0.75
VIDEO_LENGTH=30
VIDEOS_FILE_PATH=/home/pi/recordings
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
```

**Example 3: Direct Streaming to Remote Server**
```bash
OBJECT_DETECTION=False
MOTION_DETECTION=False
STREAM_IP=203.0.113.10  # Public IP
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=mySecureKey123
CAMERA_WIDTH=1920
CAMERA_HEIGHT=1080
CAMERA_FPS=30
CAMERA_AUDIO=True
```

## Deployment Options

### Option 1: Local Streaming to Media Player

Stream to VLC, Windows Media Player, or other RTMP-compatible software on the same device or local network.

**Step 1: Start Nginx Server**

```bash
docker compose up -d nginx
```

**Step 2: Configure Environment**

Update `.env`:
```bash
OBJECT_DETECTION=True  # or False for direct streaming
STREAM_IP=127.0.0.1   # for same device, or LAN IP for other devices
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=stream
```

**Step 3: Start Streaming**

```bash
./run.sh
```

**Step 4: Connect Media Player**

Open network stream in your media player:

```
rtmp://<SERVER_IP>:1935/live/stream
```

- **Same device**: `rtmp://localhost:1935/live/stream`
- **LAN device**: `rtmp://192.168.1.50:1935/live/stream` (use Pi's IP)

**Step 5: Stop Streaming**

```bash
./stop.sh
```

To stop nginx:
```bash
docker compose down
```

### Option 2: Web Browser Streaming (Local Network)

Stream to web browsers using HLS protocol.

**Step 1: Start Nginx Web Server**

```bash
docker compose up -d nginx
```

**Step 2: Configure Environment**

Update `.env`:
```bash
OBJECT_DETECTION=True
STREAM_IP=127.0.0.1  # or LAN IP of device running nginx
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=stream
```

**Step 3: Start Streaming**

On the Raspberry Pi:
```bash
./run.sh
```

**Step 4: Access Web Interface**

Open browser and navigate to:
```
http://localhost:8080/?key=stream
```

Or from another device on your network:
```
http://<PI_IP_ADDRESS>:8080/?key=stream
```

**Step 5: Stop Services**

On Raspberry Pi:
```bash
./stop.sh
```

On client machine:
```bash
docker compose down
```

### Option 3: Remote Web Server Streaming

Stream to a public web server accessible over the internet.

**Prerequisites:**
- Web server with Docker installed
- Domain name or public IP address
- Port 1935 open in firewall for RTMP

**Step 1: Clone Repository on Web Server**

```bash
ssh user@your-server.com
git clone https://github.com/yourusername/holly-stream.git
cd holly-stream
```

**Step 2: Configure Nginx for Remote Access**

Edit `nginx/nginx.conf`:

**Line 27** - Add your domain:
```nginx
server_name your-domain.com;  # Replace localhost
```

**Lines 40-43** - Whitelist your home IP:
```nginx
# Find your IP at https://whatismyipaddress.com/
allow publish 203.0.113.45;  # Your home IP address
allow publish 127.0.0.0/8;   # Keep for local testing
```

**Line 46** (optional) - Change application name:
```nginx
application live {  # Change "live" to custom name if desired
    # ...
}
```

Edit `nginx/stream/index.html`:

**Replace all instances** of `http://localhost` with your domain:
```html
<!-- Before -->
src: 'http://localhost/hls/stream.m3u8'

<!-- After -->
src: 'https://your-domain.com/hls/stream.m3u8'
```

**Line 20** - Replace stream key for security:
```html
<!-- Before -->
src: 'http://localhost/hls/stream.m3u8'

<!-- After -->
src: 'https://your-domain.com/hls/mySecureKey123.m3u8'
```

**Step 3: Start Nginx on Web Server**

```bash
docker compose up -d nginx
```

**Step 4: Configure Raspberry Pi Environment**

On Raspberry Pi, update `.env`:
```bash
OBJECT_DETECTION=True
STREAM_IP=203.0.113.10  # Your web server's public IP
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=mySecureKey123  # Match the key in index.html
```

**Step 5: Start Streaming from Raspberry Pi**

```bash
./run.sh
```

**Step 6: Access Stream**

Navigate to:
```
https://your-domain.com/index.html
```

**Security Notes:**
- Use HTTPS with SSL certificates (Let's Encrypt recommended)
- Keep stream keys private
- Whitelist only trusted IP addresses
- Consider additional authentication for production

## Advanced Configuration

### Supported Object Classes

The default YOLOv8 model detects 80 COCO classes. To detect all classes, remove the `CLASSES` variable from `.env`. To filter specific classes:

```bash
CLASSES="[0, 16, 17, 54, 67]"  # person, dog, horse, donut, cell phone
```

**Available Classes:**

| Index | Class          | Index | Class          | Index | Class          | Index | Class          |
|-------|----------------|-------|----------------|-------|----------------|-------|----------------|
| 0     | person         | 20    | elephant       | 40    | wine glass     | 60    | dining table   |
| 1     | bicycle        | 21    | bear           | 41    | cup            | 61    | toilet         |
| 2     | car            | 22    | zebra          | 42    | fork           | 62    | tv             |
| 3     | motorcycle     | 23    | giraffe        | 43    | knife          | 63    | laptop         |
| 4     | airplane       | 24    | backpack       | 44    | spoon          | 64    | mouse          |
| 5     | bus            | 25    | umbrella       | 45    | bowl           | 65    | remote         |
| 6     | train          | 26    | handbag        | 46    | banana         | 66    | keyboard       |
| 7     | truck          | 27    | tie            | 47    | apple          | 67    | cell phone     |
| 8     | boat           | 28    | suitcase       | 48    | sandwich       | 68    | microwave      |
| 9     | traffic light  | 29    | frisbee        | 49    | orange         | 69    | oven           |
| 10    | fire hydrant   | 30    | skis           | 50    | broccoli       | 70    | toaster        |
| 11    | stop sign      | 31    | snowboard      | 51    | carrot         | 71    | sink           |
| 12    | parking meter  | 32    | sports ball    | 52    | hot dog        | 72    | refrigerator   |
| 13    | bench          | 33    | kite           | 53    | pizza          | 73    | book           |
| 14    | bird           | 34    | baseball bat   | 54    | donut          | 74    | clock          |
| 15    | cat            | 35    | baseball glove | 55    | cake           | 75    | vase           |
| 16    | dog            | 36    | skateboard     | 56    | chair          | 76    | scissors       |
| 17    | horse          | 37    | surfboard      | 57    | couch          | 77    | teddy bear     |
| 18    | sheep          | 38    | tennis racket  | 58    | potted plant   | 78    | hair dryer     |
| 19    | cow            | 39    | bottle         | 59    | bed            | 79    | toothbrush     |

### Video Quality Presets

Edit `app/stream.sh` to adjust quality presets (direct streaming mode only):

```bash
# Ultra Quality (Raspberry Pi 4/5)
QUALITY="ultra"  # 10000k bitrate, 1080p, slow preset

# High Quality
QUALITY="high"   # 6000k bitrate, 1080p, medium preset

# Medium Quality (default)
QUALITY="medium" # 3000k bitrate, 720p, veryfast preset

# Low Quality (lowest latency)
QUALITY="low"    # 1500k bitrate, 720p, ultrafast preset
```

### Network Configuration

**Finding IP Addresses:**

- **Raspberry Pi** (Linux):
  ```bash
  ip a
  # Look for inet under wlan0 (WiFi) or eth0 (Ethernet)
  # Example: inet 192.168.1.50/24
  ```

- **Windows Client**:
  ```powershell
  ipconfig
  # Look for IPv4 Address
  ```

**Port Forwarding for Remote Access:**

If streaming to remote server over internet:
1. Configure router to forward port 1935 to server IP
2. Set server firewall to allow inbound TCP 1935
3. Use public IP or domain name in `STREAM_IP`

### S3 Model Repository

To host Triton models in AWS S3:

```bash
# In .env
MODEL_REPOSITORY=s3://my-bucket/triton-models/
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-west-2
```

Ensure S3 bucket structure matches:
```
my-bucket/
└── triton-models/
    ├── preprocess/
    ├── object_detection/
    └── yolo11/
```

## Custom Model Training

You train a custom model the normal Ultralytics way: a YOLO11 **detection** model fine-tuned from `yolo11n.pt`. Only two steps are specific to Holly-Stream. You collect images with the Pi's own camera (Step 1), and you convert the trained `best.pt` for Triton with `triton/export.sh` (Step 4).

### Step 1: Collect Images on the Pi

`app/collect_data.py` saves a frame every few seconds. It uses the same `rpicam-vid` capture, resolution, orientation and image tuning as object detection (all read from `.env`), so training images look exactly like what the model will see. Only one process can use the camera, so stop the app first:

```bash
docker compose stop app
docker compose run --rm --no-deps -v "$PWD/data:/root/app/data" app \
    python3 collect_data.py --count 200 --period 30
docker compose start app
sudo chown -R "$USER" data/    # the container writes the files as root
```

`collect_data.py` is baked into the app image, so this needs an image built from this version of the repo (`docker compose build app`). Images land in `data/images/<hostname>_00000.jpg`, `_00001.jpg`, ... Running it again adds to the folder without overwriting, and the hostname prefix keeps images from several Pis apart. `data/` is gitignored.

Tips for a useful dataset:
- **Spread images out over time.** Frames taken 5 seconds apart are nearly identical. A longer `--period` over a whole day covers lighting changes, and so does running it at several times of day.
- **Vary the scene.** Capture your subjects at different distances and positions, and include some images with nothing in them (around 10%).
- **Aim for a few hundred labeled instances per class to start.** More data beats more epochs.

### Step 2: Label the Images

Label in YOLO format with [CVAT](https://www.cvat.ai/), [Label Studio](https://labelstud.io/) or [Roboflow](https://roboflow.com/). Each image gets a `.txt` file with one line per object: `class x_center y_center width height`, all normalized to 0-1.

> **Label every class you want detected.** Fine-tuning replaces the 80 COCO classes with only the classes in your dataset. If you train on just `my_dog`, the model stops detecting people. To keep people, label them too.

To save time, pre-label with a large COCO model and correct the results instead of drawing every box by hand:

```bash
yolo predict model=yolo11x.pt source=data/images classes=0,16 save_txt=True   # labels go to runs/detect/predict/labels/
```

Most labeling tools can import those files. Keep in mind that the class indexes are COCO's (0=person, 16=dog), so remap them to your own class order.

Split the images about 80/20 into train and val:

```
datasets/holly/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`data.yaml`:

```yaml
path: /home/user/datasets/holly
train: images/train
val: images/val
names:
  0: person
  1: my_dog
```

### Step 3: Train

Train on a desktop, not the Pi. A CUDA GPU is strongly recommended; a CPU works for small datasets but is slow (about 1 minute per epoch for 500 images at `imgsz=320` on a 4-core i5).

```bash
pip install ultralytics onnx onnxslim    # plus a CUDA build of PyTorch, see pytorch.org

yolo detect train data=datasets/holly/data.yaml model=yolo11n.pt imgsz=320 epochs=100 batch=32 device=0
```

- **Start from `yolo11n.pt`.** The pretrained weights need far less data than training from scratch. Stay with the nano model: YOLO11s is about 3x slower on a Pi 4.
- **Train at `imgsz=320`** to match the default 256x320 export. If you export at 384x512, train at `imgsz=512`.
- **The best checkpoint** is saved to `runs/detect/train/weights/best.pt`. Check it with `yolo detect val model=runs/detect/train/weights/best.pt data=datasets/holly/data.yaml imgsz=320`.

### Step 4: Convert for Triton

```bash
./triton/export.sh runs/detect/train/weights/best.pt           # 256x320 input (default)
./triton/export.sh runs/detect/train/weights/best.pt 384 512   # larger input, see below
```

This exports ONNX with NMS built in and writes these files:
- `triton/repository/object_detection/1/model.onnx`
- `triton/repository/yolo11/labels.txt`, generated from the model's class names
- the input size in both `config.pbtxt` files

It prints the class indexes at the end. Set `CLASSES` in `.env` to match (for example `CLASSES="[0, 1]"`), or `CLASSES="[]"` to show every class.

Input size is a speed/accuracy trade-off. Measured on a Pi 4 while streaming 1280x960@30, with stock YOLO11n accuracy on COCO val2017:

| Input (HxW) | Inference | Person AP50-95 | Small / medium / large people |
|-------------|-----------|----------------|-------------------------------|
| 256x320 (default) | ~210 ms | 39.2 | 12.3 / 46.7 / 74.1 |
| 384x512 | ~500 ms | 48.5 | 23.1 / 58.4 / 77.9 |
| 640x640 | ~1000 ms | 51.8 | 28.9 / 62.2 / 78.1 |

256x320 works well when subjects fill a reasonable part of the frame, such as a room. Use 384x512 if you need to catch small or distant subjects, such as across a yard.

The model drops detections below 0.25 confidence inside the graph, so `CONFIDENCE_THRESHOLD` can only raise that floor. To go lower, export with `CONF=0.1 ./triton/export.sh ...`.

### Step 5: Deploy

Get the updated `triton/repository` onto each Pi, either by committing and pulling or with `scp -r triton/repository <pi>:holly-stream/triton/`. Then restart:

```bash
docker compose restart triton app
```

### Step 6: Test

From the machine you exported on (`test.py` reads `labels.txt` from its local repository), send one image through the Pi's Triton:

```bash
python triton/test.py --url grpc://<pi-ip>:8001 --image data/images/rustypi6_00012.jpg --output results.png
```

Then watch the stream. If Triton fails to load the model, run `docker logs holly-stream-triton`; the usual cause is a `config.pbtxt` edited by hand to a size that doesn't match the exported model.

## Multi-Camera Setup

Manage multiple Raspberry Pi cameras from a central location.

### Configuration

On your control machine, edit `.env`:

```bash
# Multi-camera orchestration
CAMERA_USERS="pi,pi,pi"
CAMERA_HOSTNAMES="192.168.1.10,192.168.1.11,192.168.1.12"
CAMERA_REPO_PATHS="/home/pi/holly-stream,/home/pi/holly-stream,/home/pi/holly-stream"
```

**Requirements:**
- SSH access to all camera devices
- Holly-stream installed on each device
- Each device has its own `.env` configuration

### Starting All Cameras

```bash
./run-all-cameras.sh
```

This script will:
1. SSH into each camera host
2. Navigate to repository path
3. Execute `./run.sh`
4. Start streaming on all devices simultaneously

### Stopping All Cameras

```bash
./stop-all-cameras.sh
```

### Use Cases

- **Multi-angle surveillance**: Monitor different areas
- **Event streaming**: Multiple camera views of same event
- **Redundancy**: Backup cameras for critical monitoring
- **Distributed detection**: Each camera detects different objects

## Troubleshooting

### Common Issues

**Issue: Camera not detected**

```bash
# Check camera connection
libcamera-hello  # Modern cameras
raspistill -t 0  # Legacy cameras
v4l2-ctl --list-devices  # USB cameras

# Verify device exists
ls /dev/video*
```

**Solution:**
- Ensure camera is properly connected
- Enable camera in `raspi-config`
- Install required drivers (ArduCam users)
- Try different `CAMERA_BACKEND` values in `.env`

---

**Issue: Triton container fails health check**

```bash
# Check Triton logs
docker logs holly-stream-triton

# Test Triton endpoint
curl http://localhost:8000/v2/health/ready
```

**Solution:**
- Verify model files exist in `triton/object_detection/1/model.onnx`
- Check `config.pbtxt` for correct input/output dimensions
- Ensure sufficient memory (4GB+ RAM)
- Verify model repository path in `.env`

---

**Issue: FFmpeg encoding errors**

```bash
# Check app container logs
docker logs holly-stream-app

# Look for FFmpeg errors
docker logs holly-stream-app 2>&1 | grep -i error
```

**Solution:**
- Verify `STREAM_IP` and `STREAM_PORT` are correct
- Ensure nginx is running and accepting connections
- Check network connectivity between containers
- Try lower resolution or FPS settings

---

**Issue: High CPU usage or low FPS**

**Solutions:**
- Reduce `CAMERA_FPS` to 15 or 20
- Lower `CAMERA_WIDTH` and `CAMERA_HEIGHT` to 640x480
- Increase inference period in `app/main.py` (change `period = 10` to higher value)
- Disable object detection: `OBJECT_DETECTION=False`
- Use hardware encoding if available

---

**Issue: No video in web browser**

```bash
# Check nginx HLS output
docker exec holly-stream-nginx ls -la /var/www/html/stream/hls/

# Should see .m3u8 and .ts files
```

**Solution:**
- Verify the nginx container is running
- Check browser console for JavaScript errors
- Ensure `STREAM_KEY` matches filename in HLS directory
- Clear browser cache
- Try different browser (Chrome, Firefox)

---

**Issue: Motion detection not triggering**

**Solution:**
- Lower `MOTION_THRESHOLD` (try 0.5 for higher sensitivity)
- Ensure adequate lighting (poor lighting causes false positives)
- Verify camera is not obstructed
- Check `docker logs holly-stream-app` for SSIM scores

---

**Issue: SCP upload fails (motion detection)**

```bash
# Test SCP manually
scp test.mp4 user@192.168.1.100:/path/to/videos/

# Check SSH key authentication
ssh user@192.168.1.100
```

**Solution:**
- Set up SSH key authentication (password-less login)
- Verify `STREAM_USER`, `STREAM_IP`, and `VIDEOS_FILE_PATH`
- Ensure remote directory exists and is writable
- Check network connectivity

---

**Issue: Docker permission errors**

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Reboot to apply
sudo reboot
```

---

### Performance Optimization

**For Raspberry Pi 4:**
- Use 720p resolution maximum
- Set FPS to 20-30
- Enable hardware encoding (automatic in `stream.sh`)
- Allocate 256MB GPU memory in `/boot/config.txt`:
  ```
  gpu_mem=256
  ```

**For Raspberry Pi 5:**
- Can handle 1080p at 30 FPS
- Object detection performs better than Pi 4
- No GPU memory allocation needed

**General Tips:**
- Use wired Ethernet for more stable streaming
- Keep Raspberry Pi cool (heatsinks, fan)
- Use high-quality power supply (5V 3A minimum)
- Close unnecessary background processes

---

### Logs and Debugging

**View container logs:**

```bash
# App container
docker logs -f holly-stream-app

# Triton container
docker logs -f holly-stream-triton

# Nginx container
docker logs -f holly-stream-nginx
```

**Check container health:**

```bash
docker ps
docker inspect holly-stream-app
docker stats
```

**Test network connectivity:**

```bash
# Test RTMP connection
ffmpeg -i rtmp://localhost:1935/live/stream -frames:v 1 test.jpg

# Test Triton
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/models/yolov8n
```

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [Nvidia Triton Inference Server](https://github.com/triton-inference-server/server) - Model serving
- [FFmpeg](https://ffmpeg.org/) - Video encoding
- [Nginx RTMP Module](https://github.com/arut/nginx-rtmp-module) - Streaming server
- [Picamera2](https://github.com/raspberrypi/picamera2) - Raspberry Pi camera interface

---

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section above
