#!/bin/bash
RTMP_URI=location=rtmp://"$STREAM_IP":"$STREAM_PORT"/"$STREAM_APPLICATION"/"$STREAM_KEY"" live=true"
gst-launch-1.0 nvarguscamerasrc sensor-id=$CAMERA_INDEX ! \
'video/x-raw(memory:NVMM)', width=$CAMERA_WIDTH, height=$CAMERA_HEIGHT, framerate=$CAMERA_FPS/1, format=NV12 ! \
nvv4l2h264enc ! \
h264parse ! \
flvmux ! \
rtmpsink "$RTMP_URI"