#!/bin/bash

DIMS="$CAMERA_WIDTH"x"$CAMERA_HEIGHT"

# video only
ffmpeg \
-f v4l2 \
-input_format mjpeg \
-video_size $DIMS \
-i /dev/video$CAMERA_INDEX \
-r $CAMERA_FPS \
-c:v libx264 \
-preset fast \
-pix_fmt yuv420p \
-b:v 1500k \
-maxrate 1500k \
-bufsize 3000k \
-g 60 \
-f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY

# audio/video
#ffmpeg \
#-f v4l2 \
#-i /dev/video$CAMERA_INDEX \
#-f alsa \
#-i default \
#-c:v libx264 \
#-preset fast \
#-pix_fmt yuv420p \
#-b:v 1500k \
#-maxrate 1500k \
#-bufsize 3000k \
#-g 60 \
#-c:a aac \
#-b:a 128k \
#-ac 2 \
#-f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY
