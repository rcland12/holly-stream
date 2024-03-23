#!/bin/bash
libcamera-still -t 5000 -n -o test.jpg
# libcamera-vid \
# --framerate $CAMERA_FPS \
# --nopreview \
# --inline \
# -t 0 \
# --width $CAMERA_WIDTH \
# --height $CAMERA_HEIGHT \
# --rotation 180 \
# --listen -o - | \
# ffmpeg \
# -i - \
# -profile:v high \
# -pix_fmt yuvj420p \
# -level:v 4.1 \
# -preset ultrafast \
# -tune zerolatency \
# -vcodec libx264 \
# -r $CAMERA_FPS \
# -s "$CAMERA_WIDTH"x"$CAMERA_HEIGHT" \
# -f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY