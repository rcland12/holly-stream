#!/bin/bash
xhost +

docker run --rm \
--interactive \
--tty \
--net=host \
--env DISPLAY=$DISPLAY \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--volume /tmp/argus_socket:/tmp/argus_socket \
--volume ${STREAM_PATH}/weights:/root/app/weights \
jetson-stream:latest
