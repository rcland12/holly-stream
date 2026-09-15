#!/bin/bash

source .env

num_cameras=${#CAMERA_USERS[@]}

for i in ${!CAMERA_USERS[@]}; do
    ssh -f ${CAMERA_USERS[$i]}@${CAMERA_HOSTNAMES[$i]} "cd ${CAMERA_REPO_PATHS[$i]} && ./stop.sh > /dev/null 2>&1"
done

exit 0
