#!/bin/bash
xhost +

docker run --rm \
--interactive \
--tty \
--net=host \
--env DISPLAY=$DISPLAY \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--volume /tmp/argus_socket:/tmp/argus_socket \
--volume /home/russ/holly-stream/weights:/root/app/weights \
jetson-stream:latest
