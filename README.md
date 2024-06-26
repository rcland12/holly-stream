<img src="./logo.png" alt="Failed to load image." style="width: auto;">
This application will ingest your computers webcam feed (using ffmpeg), apply an object detection task on the feed with bounding boxes, and send that feed via RTMP to an address of your choice. You can also turn off object detection to create a simple live stream camera, good for a security system or monitoring system.

---

This project has three main branches, [`jetson`](https://github.com/rcland12/holly-stream/tree/jetson), [`linux`](https://github.com/rcland12/holly-stream/tree/linux), and [`raspbian`](https://github.com/rcland12/holly-stream/tree/raspbian). The `jetson` branch is intended for use on Nvidia Jetson architecture (JetPack OS). The `linux` branch is intended for use on a Linux-based operating system (Debian, Arch, etc.). The `raspbian` branch is intended for use on a Raspberry Pi machine (tested on a Raspberry Pi 4, Raspian OS). Switch to the corresponding branch to get started.

### Jetson
```bash
# https
git clone --branch jetson --depth 1 https://github.com/rcland12/holly-stream.git

# ssh
git clone --branch jetson --depth 1 git@github.com:rcland12/holly-stream.git
```

### Linux
```bash
# https
git clone --branch linux --depth 1 https://github.com/rcland12/holly-stream.git

# ssh
git clone --branch linux --depth 1 git@github.com:rcland12/holly-stream.git
```

### Raspbian
```bash
# https
git clone --branch raspbian --depth 1 https://github.com/rcland12/holly-stream.git

# ssh
git clone --branch raspbian --depth 1 git@github.com:rcland12/holly-stream.git
```

# How to Contribute
You are welcome to contribute to this repository. Simply fork this repository and submit pull requests for my review. Here are areas this project can be improved upon:

* Adding more supported OS and architectures. Each branch was tested on the following hardware:
    * Jetson: Jetson Nano with Jetpack 4.6.4 and CSI camera (IMX219-160 8MP)
    * Linux: Ubuntu 22.04 with Nvidia GTX 1060 6GB and USB camera (Logitech 1080p)
    * Raspbian: Raspberry Pi 4 with Raspbian 64-bit (Debian bookworm) and CSI camera (IMX519 16MP)
* Optimizing streaming latency through alternative protocols (WebRTC, Dash, udp)
* Overall code optimization
* How to deploy each app, the raspbian stream app cannot be used in Docker so for now it is launched with a bash script.