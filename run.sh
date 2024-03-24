#!/bin/bash
source .env

if [ -z $OBJECT_DETECTION ]; then echo "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False."; fi

ATTEMPT=1
MAX_RETRY=6
RETRY_TIME=10

if [ "${OBJECT_DETECTION}" == "True" ]; then
    docker compose up -d triton
    while [ $ATTEMPT -le $MAX_RETRY ]; do
        url="http://localhost:8000/v2/health/ready"
        response=$(curl --write-out "%{http_code}" --silent --output /dev/null "$url")

        if [ $response -eq 200 ]; then
            echo "Triton is healthy, starting holly-stream."
            break
        else
            echo "Triton is not healthy yet, trying again in $RETRY_TIME seconds. Attempt ($ATTEMPT/$MAX_RETRY)"
            ATTEMPT=$((ATTEMPT + 1))
            sleep $RETRY_TIME
        fi
    done

    if [ $ATTEMPT -gt $MAX_RETRY ]; then
        echo "Command failed after $MAX_RETRY attempts"
    fi

    python3 app/main.py
    # echo $! > .process.pid

elif [ "${OBJECT_DETECTION}" == "False" ]; then
    # Checking if required environment variables are defined. If not, then defining them.
    if [ -z $CAMERA_FPS ]; then echo "You did not define CAMERA_FPS. It will default to 30." && CAMERA_FPS=30; fi
    if [ -z $CAMERA_WIDTH ]; then echo "You did not define CAMERA_WIDTH. It will default to 1280." && CAMERA_WIDTH=1280; fi
    if [ -z $CAMERA_HEIGHT ]; then echo "You did not define CAMERA_HEIGHT. It will default to 720." && CAMERA_HEIGHT=720; fi
    if [ -z $STREAM_IP ]; then echo "You did not define STREAM_IP. It will default to 127.0.0.1. This is the IP address where the RTMP stream will be sent." && STREAM_IP=127.0.0.1; fi
    if [ -z $STREAM_PORT ]; then echo "You did not define STREAM_PORT. It will default to 1935." && STREAM_PORT=1935; fi
    if [ -z $STREAM_APPLICATION ]; then echo "You did not define STREAM_APPLICATION. It will default to 'live'. If you are using Nginx as the RTMP server, this has to match the application name in the config." && STREAM_APPLICATION=live; fi
    if [ -z $STREAM_KEY ]; then echo "You did not define STREAM_KEY. It will default to 'stream'. If you are using HLS and a website to stream, this must match the name of your .m3u8 file." && STREAM_KEY=stream; fi

    libcamera-vid \
        --nopreview \
        --inline \
        --timeout 0 \
        --framerate $CAMERA_FPS \
        --width $CAMERA_WIDTH \
        --height $CAMERA_HEIGHT \
        --rotation 180 \
        --listen -o - | \
    ffmpeg \
        -i - \
        -nostdin \
        -profile:v high \
        -pix_fmt yuvj420p \
        -level:v 4.1 \
        -preset ultrafast \
        -tune zerolatency \
        -vcodec libx264 \
        -r $CAMERA_FPS \
        -s "$CAMERA_WIDTH"x"$CAMERA_HEIGHT" \
        -f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY \
        > /dev/null 2>&1 &
    echo $! > .process.pid

else
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
    exit 120
fi
