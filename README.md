# Holly-Stream

<img src="./app/images/logo.png" alt="Holly-Stream Logo" style="width: auto;">

A containerized live streaming application for Ubuntu servers featuring real-time object detection and multi-protocol streaming capabilities. Built with YOLOv8/YOLOv11, NVIDIA Triton Inference Server, and FFmpeg for high-performance GPU-accelerated video processing.

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

Holly-Stream transforms your Ubuntu server into a powerful streaming platform capable of:

- **Real-time Object Detection**: Using YOLOv8/YOLOv11 models with NVIDIA Triton Inference Server and TensorRT optimization for maximum GPU performance
- **Live Streaming**: Direct camera feed streaming with minimal latency and hardware-accelerated encoding
- **Multi-Protocol Support**: Stream via RTMP to media players (VLC, OBS) or HLS for web browsers
- **Flexible Deployment**: Local network, remote web server, or localhost streaming options
- **Adaptive Performance**: Automatic frame-skip adjustment based on inference latency to maintain consistent streaming quality

## Features

### Core Capabilities

- **Object Detection Pipeline**

  - YOLOv8/YOLOv11 model inference with TensorRT optimization
  - 80 COCO classes supported by default (customizable)
  - Configurable confidence and IOU thresholds
  - Real-time bounding box annotation
  - Selective class filtering
  - GPU-accelerated inference with NVIDIA Triton Inference Server

- **Camera Support**

  - USB webcams via V4L2
  - Configurable resolution and framerate
  - Audio capture from USB microphones or built-in sources via ALSA/PulseAudio
  - Multiple camera device support

- **Streaming Protocols**

  - RTMP streaming for media players (VLC, OBS, Windows Media Player)
  - HLS streaming for web browsers
  - Configurable quality presets (ultra/high/medium/low/fast)
  - Hardware-accelerated H.264 encoding (NVENC when available)
  - Low-latency zero-latency tuning options

- **Additional Features**
  - Santa Hat Plugin (novelty overlay for detected objects)
  - Multi-camera orchestration and remote management
  - Data collection utility for custom model training
  - Automatic adaptive inference performance
  - S3 model repository support for centralized model management
  - Docker containerized for easy deployment

## Architecture

### System Overview

```mermaid
graph TB
    subgraph UbuntuServer["Ubuntu Server Host"]
        Camera["Camera Device<br/>/dev/video0"]

        subgraph DockerEnv["Docker Environment"]
            subgraph AppContainer["App Container"]
                OpenCV["OpenCV<br/>Frame Capture"]
                Detector["Detection Logic"]
                Annotator["Frame Annotation"]
                FFmpeg["FFmpeg Encoder<br/>H.264/RTMP"]
            end

            subgraph TritonContainer["Triton Container"]
                Preprocess["Preprocess Model<br/>Letterbox + Normalize"]
                YOLOModel["YOLOv8/v11 TensorRT<br/>GPU Detection"]
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
        MediaPlayer["Media Players<br/>VLC, OBS, WMP"]
        WebBrowser["Web Browsers<br/>Video.js Player"]
    end

    Camera -->|Video Feed| OpenCV
    OpenCV -->|Raw Frames| Detector
    Detector <-->|gRPC| Preprocess
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
    style UbuntuServer fill:#f5f5f5
```

### Data Flow

**Object Detection Mode:**

1. Camera captures frame at configured FPS (e.g., 30 FPS)
2. Adaptive frame skip (1-6x) sends frames to Triton Inference Server based on GPU performance
3. Triton pipeline: Preprocess → Detection (TensorRT) → Postprocess (NMS)
4. Detection results returned to application via gRPC
5. Frame annotated with bounding boxes and labels
6. All frames encoded with FFmpeg and streamed via RTMP
7. Nginx receives RTMP stream and converts to HLS
8. Clients connect via RTMP (direct) or HTTP/HLS (web browsers)

**Direct Streaming Mode:**

1. Camera feed captured directly with OpenCV
2. Frames piped to FFmpeg encoder
3. H.264 encoding with minimal latency (no detection overhead)
4. RTMP output to nginx server
5. Lower CPU/GPU usage for basic streaming needs

## System Requirements

### Hardware

- **Server**: Ubuntu 20.04+ capable machine
- **Memory**: 4GB RAM minimum (8GB+ recommended for object detection)
- **Camera**: USB webcam, 720p minimum, 1080p recommended
- **Storage**: 10GB+ free disk space
- **Network**: Ethernet or WiFi connection

### Software

- **Operating System**: Ubuntu 20.04+ (Linux)
- **Docker**: Version 20.10 or later with compose plugin
- **FFmpeg**: Version 4.0 or later (included in Docker container)

### Optional Requirements

- **For Object Detection (GPU-Accelerated)**:
  - NVIDIA CUDA-enabled GPU (GTX 1060 6GB or better recommended)
  - CUDA Version 11.8+ (handled by Docker container)
  - GPU Memory: 2GB VRAM minimum (4GB+ recommended)
  - nvidia-docker2 runtime installed
- **For Custom Model Training**: CUDA-enabled machine for training (not required for inference)
- **For S3 Model Repository**: AWS account with S3 access

**Note**: Object detection can be disabled for basic streaming without a GPU. CPU-only inference is not recommended due to performance limitations.

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/holly-stream.git
cd holly-stream
```

### 2. Install NVIDIA Docker Runtime (for GPU support)

If using object detection with GPU:

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

### 3. Pull Docker Images

Pull the pre-built container images:

```bash
# For object detection (GPU required)
docker pull rcland12/detection-stream:linux-triton-latest

# For nginx streaming server
docker pull rcland12/detection-stream:nginx-latest

# For main application
docker pull rcland12/detection-stream:linux-latest
```

### 4. Make Scripts Executable

```bash
chmod +x ./run.sh ./stop.sh
```

### 5. Camera Setup

**For USB Webcam:**

- Connect camera to USB port
- Verify device: `ls /dev/video*`
- Note device number (usually `/dev/video0`)
- Test camera:
  ```bash
  ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg
  ```

**For Multiple Cameras:**

- List all video devices: `v4l2-ctl --list-devices`
- Each camera will have a different index (video0, video1, etc.)

## Configuration

### Environment Variables

Create a `.env` file in the project root directory. This file controls all operational parameters.

**Required Variables:**

```bash
# Operation Mode (required)
OBJECT_DETECTION=False  # True/False - enables/disables object detection
```

**Camera Configuration:**

```bash
# Camera Settings
CAMERA_WIDTH=1280           # Resolution width in pixels
CAMERA_HEIGHT=720           # Resolution height in pixels
CAMERA_FPS=30               # Frames per second
CAMERA_INDEX=0              # USB camera device index (/dev/videoX)
```

**Streaming Configuration:**

```bash
# RTMP Stream Settings
STREAM_IP=127.0.0.1         # Target server IP address
STREAM_PORT=1935            # RTMP port (default: 1935)
STREAM_APPLICATION=live     # RTMP application name
STREAM_KEY=stream           # RTMP stream key
STREAM_QUALITY=fast         # Encoding preset (ultra/high/medium/low/fast)
```

**Audio Configuration:**

```bash
# Audio Settings
AUDIO_ENABLED=True          # Include audio in stream (True/False)
AUDIO_DEVICE=hw:Webcam,0    # ALSA device or pulse:default for PulseAudio
```

**Object Detection Settings:**

```bash
# Model Configuration
TRITON_URL=grpc://holly-stream-triton:8001  # Triton server address
MODEL_NAME=yolo11                            # Model identifier in Triton
MODEL_DIMS=(640, 640)                        # Model input dimensions (tuple)
MODEL_REPOSITORY=/root/app/triton            # Path to Triton model repository
CLASSES=[]                                   # Filter specific classes (empty = all 80)

# AWS S3 Model Repository (optional)
AWS_ACCESS_KEY_ID=your_key_id
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

**Special Features:**

```bash
# Santa Hat Plugin
SANTA_HAT_PLUGIN=False      # Overlay santa hat on highest confidence detection
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
CLASSES=[0, 16]  # Detect only persons and dogs
```

**Example 2: Direct Streaming to Remote Server**

```bash
OBJECT_DETECTION=False
STREAM_IP=203.0.113.10  # Public IP
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=mySecureKey123
CAMERA_WIDTH=1920
CAMERA_HEIGHT=1080
CAMERA_FPS=30
AUDIO_ENABLED=True
```

**Example 3: High-Performance GPU Detection**

```bash
OBJECT_DETECTION=True
STREAM_IP=192.168.1.100  # LAN device
STREAM_PORT=1935
STREAM_KEY=stream
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30
STREAM_QUALITY=high
CLASSES=[0, 1, 2, 3, 5, 7]  # Vehicles and people
```

## Deployment Options

### Option 1: Local Streaming to Media Player

Stream to VLC, OBS, or other RTMP-compatible software on the same device or local network.

**Step 1: Start Nginx Server**

```bash
docker compose up -d nginx-stream
```

**Step 2: Configure Environment**

Update `.env`:

```bash
OBJECT_DETECTION=True  # or False for direct streaming
STREAM_IP=127.0.0.1    # for same device, or LAN IP for other devices
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
- **LAN device**: `rtmp://192.168.1.50:1935/live/stream` (use server's IP)

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
docker compose up -d nginx-web
```

**Step 2: Configure Environment**

Update `.env`:

```bash
OBJECT_DETECTION=True
STREAM_IP=127.0.0.1  # or LAN IP of device running nginx-web
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=stream
```

**Step 3: Start Streaming**

On the Ubuntu server:

```bash
./run.sh
```

**Step 4: Access Web Interface**

Open browser and navigate to:

```
http://localhost:8080/index.html
```

Or from another device on your network:

```
http://<SERVER_IP>:8080/index.html
```

**Step 5: Stop Services**

On Ubuntu server:

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

Edit `nginx/nginx-web/nginx.conf`:

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
docker compose up -d nginx-web
```

**Step 4: Configure Ubuntu Server Environment**

On Ubuntu server, update `.env`:

```bash
OBJECT_DETECTION=True
STREAM_IP=203.0.113.10  # Your web server's public IP
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=mySecureKey123  # Match the key in index.html
```

**Step 5: Start Streaming from Ubuntu Server**

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

## Custom Model Training

Train a custom YOLOv8/YOLOv11 model for specific objects (e.g., detect your pet, specific vehicles, custom objects).

**Important**: TensorRT model files (`.plan`) are machine-specific. A model converted on one GPU architecture will not work on another. You must convert models on your target deployment hardware.

### Step 1: Data Collection

Use the provided data collection utility:

```bash
python3 app/collect_data.py
```

This will capture images at intervals for dataset creation. Collect 200-500 images per class minimum.

### Step 2: Data Annotation

Annotate images in YOLO format. Recommended tools:

- [Roboflow](https://roboflow.com/) (online, user-friendly)
- [LabelImg](https://github.com/tzutalin/labelImg) (offline)

**Dataset Split:**

- Training: 70-80%
- Validation: 20-30%
- Testing: Optional 10% holdout

### Step 3: Training Environment Setup

On a CUDA-enabled machine (local or cloud):

```bash
pip install ultralytics torch torchvision
```

### Step 4: Train Model

Create a training script `train.py`:

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # nano model for speed

# Train
model.train(
    data='data.yaml',        # Path to dataset configuration
    epochs=100,              # Training epochs (increase for better accuracy)
    batch=16,                # Batch size (adjust for GPU memory)
    imgsz=640,               # Image size
    device=0,                # GPU device (0 for first GPU, 'cpu' for CPU)
    optimizer='AdamW',       # Optimizer
    patience=50,             # Early stopping patience
    save=True,               # Save checkpoints
    project='runs/detect',   # Output directory
    name='custom_model'      # Experiment name
)
```

Create `data.yaml`:

```yaml
# Paths (use absolute paths)
train: /home/user/holly-stream/datasets/train/images
val: /home/user/holly-stream/datasets/val/images

# Classes
nc: 2  # Number of classes
names: ["my_dog", "my_cat"]  # Class names
```

Run training:

```bash
python3 train.py
```

Training time varies: 2-6 hours on RTX 3060, 12-24+ hours on CPU.

### Step 5: Export to TensorRT

After training completes:

```python
from ultralytics import YOLO

# Load best model
model = YOLO('runs/detect/custom_model/weights/best.pt')

# Export to ONNX first
model.export(
    format='onnx',      # ONNX format
    half=True,          # FP16 quantization (smaller, faster)
    simplify=True,      # Simplify graph
    opset=12            # ONNX opset version
)
```

Convert ONNX to TensorRT on your Ubuntu server:

```bash
docker run -it --rm --gpus all -v ./runs/detect/custom_model/weights:/models \
    nvcr.io/nvidia/tensorrt:23.12-py3 \
    trtexec --onnx=/models/best.onnx \
            --saveEngine=/models/best.plan \
            --fp16 \
            --inputIOFormats=fp16:chw \
            --outputIOFormats=fp16:chw
```

This process takes approximately 2-5 minutes.

### Step 6: Deploy Custom Model

**Copy model to Triton repository:**

```bash
cp runs/detect/custom_model/weights/best.plan triton/object_detection/1/model.plan
```

**Update Triton configuration** (if model dimensions differ):

Edit `triton/object_detection/config.pbtxt`:

```protobuf
input [
  {
    name: "images"
    data_type: TYPE_FP16
    dims: [ 1, 3, 640, 640 ]  # Match your model input
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP16
    dims: [ 1, 84, 8400 ]  # Verify with model.info()
  }
]
```

**Update class labels:**

Edit `triton/yolo11/labels.txt`:

```
my_dog
my_cat
```

**Update environment variables:**

```bash
MODEL_NAME=yolo11
MODEL_DIMS=(640, 640)  # Match training imgsz
CLASSES=[0, 1]         # All custom classes
```

### Step 7: Test Custom Model

```bash
./run.sh
```

Monitor logs for inference errors. Adjust confidence threshold as needed in the code or via additional configuration.

## Advanced Configuration

### Supported Object Classes

The default YOLOv8/YOLOv11 model detects 80 COCO classes. To detect all classes, remove the `CLASSES` variable from `.env`. To filter specific classes:

```bash
CLASSES=[0, 16, 17, 54, 67]  # person, dog, horse, donut, cell phone
```

**Available Classes:**

| Index | Class         | Index | Class          | Index | Class        | Index | Class        |
| ----- | ------------- | ----- | -------------- | ----- | ------------ | ----- | ------------ |
| 0     | person        | 20    | elephant       | 40    | wine glass   | 60    | dining table |
| 1     | bicycle       | 21    | bear           | 41    | cup          | 61    | toilet       |
| 2     | car           | 22    | zebra          | 42    | fork         | 62    | tv           |
| 3     | motorcycle    | 23    | giraffe        | 43    | knife        | 63    | laptop       |
| 4     | airplane      | 24    | backpack       | 44    | spoon        | 64    | mouse        |
| 5     | bus           | 25    | umbrella       | 45    | bowl         | 65    | remote       |
| 6     | train         | 26    | handbag        | 46    | banana       | 66    | keyboard     |
| 7     | truck         | 27    | tie            | 47    | apple        | 67    | cell phone   |
| 8     | boat          | 28    | suitcase       | 48    | sandwich     | 68    | microwave    |
| 9     | traffic light | 29    | frisbee        | 49    | orange       | 69    | oven         |
| 10    | fire hydrant  | 30    | skis           | 50    | broccoli     | 70    | toaster      |
| 11    | stop sign     | 31    | snowboard      | 51    | carrot       | 71    | sink         |
| 12    | parking meter | 32    | sports ball    | 52    | hot dog      | 72    | refrigerator |
| 13    | bench         | 33    | kite           | 53    | pizza        | 73    | book         |
| 14    | bird          | 34    | baseball bat   | 54    | donut        | 74    | clock        |
| 15    | cat           | 35    | baseball glove | 55    | cake         | 75    | vase         |
| 16    | dog           | 36    | skateboard     | 56    | chair        | 76    | scissors     |
| 17    | horse         | 37    | surfboard      | 57    | couch        | 77    | teddy bear   |
| 18    | sheep         | 38    | tennis racket  | 58    | potted plant | 78    | hair dryer   |
| 19    | cow           | 39    | bottle         | 59    | bed          | 79    | toothbrush   |

### Video Quality Presets

Stream quality is configured via `STREAM_QUALITY` in `.env`:

```bash
# Ultra Quality (for powerful GPUs)
STREAM_QUALITY=ultra  # High bitrate, best quality, slower encoding

# High Quality
STREAM_QUALITY=high   # Balanced quality and performance

# Medium Quality (default)
STREAM_QUALITY=medium # Good quality, fast encoding

# Low Quality
STREAM_QUALITY=low    # Lowest latency, minimum bandwidth

# Fast (lowest latency)
STREAM_QUALITY=fast   # Optimized for real-time streaming
```

### Network Configuration

**Finding IP Addresses:**

- **Ubuntu Server** (Linux):

  ```bash
  ip a
  # Look for inet under eth0 (Ethernet) or wlan0 (WiFi)
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

```bash
# Open firewall port on Ubuntu
sudo ufw allow 1935/tcp
sudo ufw status
```

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
    ├── postprocess/
    └── yolo11/
```

## Troubleshooting

### Common Issues

**Issue: Camera not detected**

```bash
# Check camera connection
ls -l /dev/video*

# Test camera
ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg
```

**Solution:**

- Ensure camera is properly connected
- Update `CAMERA_INDEX` in `.env` to match your device
- Try different USB ports
- Check USB permissions: `sudo chmod 666 /dev/video0`

---

**Issue: Triton container fails health check**

```bash
# Check Triton logs
docker logs holly-stream-triton

# Test Triton endpoint
curl http://localhost:8000/v2/health/ready
```

**Solution:**

- Verify model files exist in `triton/object_detection/1/model.plan`
- Check `config.pbtxt` for correct input/output dimensions
- Ensure sufficient GPU memory (4GB+ VRAM)
- Verify model was converted on the same GPU architecture
- Check nvidia-docker runtime is installed

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
- Disable object detection: `OBJECT_DETECTION=False`
- Use faster encoding preset: `STREAM_QUALITY=fast`
- Enable hardware encoding if available (NVENC)

---

**Issue: CUDA out of memory errors**

```bash
# Check GPU memory usage
nvidia-smi
```

**Solutions:**

- Close other GPU applications
- Use smaller model (YOLOv8n instead of YOLOv8m/l/x)
- Reduce camera resolution
- Check for memory leaks in logs

---

**Issue: Stream not connecting**

**For localhost:**

```bash
# Check if port is available
sudo netstat -tulpn | grep 1935
```

**For remote streaming:**

```bash
# Verify firewall rules
sudo ufw status
sudo ufw allow 1935/tcp
```

**Solution:**

- Ensure nginx container is running: `docker ps`
- Verify correct IP and port in `.env`
- Check firewall/router settings
- Test connectivity: `telnet <STREAM_IP> 1935`

---

**Issue: No video in web browser**

```bash
# Check nginx HLS output
docker exec holly-stream-nginx-web-1 ls -la /var/www/html/stream/hls/

# Should see .m3u8 and .ts files
```

**Solution:**

- Verify nginx-web container is running
- Check browser console for JavaScript errors
- Ensure `STREAM_KEY` matches filename in HLS directory
- Clear browser cache
- Try different browser (Chrome, Firefox)

---

**Issue: Audio not working**

```bash
# List audio devices
arecord -l

# Test audio capture
arecord -D hw:Webcam,0 -d 5 test.wav
```

**Solution:**

- Update `AUDIO_DEVICE` with correct hardware identifier
- Check audio permissions
- Try PulseAudio: `AUDIO_DEVICE=pulse:default`
- Verify microphone is not muted

---

**Issue: Docker permission errors**

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Reboot to apply
sudo reboot
```

---

### Logs and Debugging

**View container logs:**

```bash
# App container
docker logs -f holly-stream-app

# Triton container
docker logs -f holly-stream-triton

# Nginx container
docker logs -f holly-stream-nginx-web-1
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
curl http://localhost:8000/v2/models/yolo11
```

---

## Multi-Camera Setup

Manage multiple Ubuntu server cameras from a central location.

### Configuration

On your control machine, edit `.env`:

```bash
# Multi-camera orchestration
CAMERA_USERS="user,user,user"
CAMERA_HOSTNAMES="192.168.1.10,192.168.1.11,192.168.1.12"
CAMERA_REPO_PATHS="/home/user/holly-stream,/home/user/holly-stream,/home/user/holly-stream"
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

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

---

## Acknowledgments

- [Ultralytics YOLOv8/YOLOv11](https://github.com/ultralytics/ultralytics) - Object detection framework
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) - Model serving
- [FFmpeg](https://ffmpeg.org/) - Video encoding
- [Nginx RTMP Module](https://github.com/arut/nginx-rtmp-module) - Streaming server
- [Video.js](https://videojs.com/) - HTML5 video player

---

## Support

For questions, issues, or feature requests:

- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section above
