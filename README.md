# Holly Stream
This application will ingest your computers webcam feed (using ffmpeg), apply an object detection task on the feed with bounding boxes, and send that feed via RTMP to an address of your choice. You have the following options for recording and applying a custom object detection model:

This project has two main branches, `jetson` and `linux`. The `jetson` branch is intended for use on a Nvidia Jetson Nano. The `linux` branch is intended for use on a Linux-based operating system (Debian, Arch, etc.). Switch to the corresponding branch to get started.

### Jetson Nano
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

# How to Contribute
