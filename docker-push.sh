#!/bin/bash
set -a
source .env
set +a

if [[ -z "${DOCKER_USERNAME}" ]]; then
    echo "Set your DOCKER_USERNAME in your .env file"
    exit 1
fi

if [[ -z "${DOCKER_PASSWORD}" ]]; then
    echo "Set your DOCKER_PASSWORD in your .env file"
    exit 1
fi

if [[ -z "${LATEST_VERSION}" ]]; then
    echo "Set the LATEST_VERSION in your .env file"
    exit 1
fi

docker login -u ${DOCKER_USERNAME} -p ${DOCKER_PASSWORD}
docker images -q rcland12/detection-stream:jetson-latest | xargs -I{} docker tag {} rcland12/detection-stream:jetson-${LATEST_VERSION}
docker images -q rcland12/detection-stream:jetson-triton-latest | xargs -I{} docker tag {} rcland12/detection-stream:jetson-triton-${LATEST_VERSION}
docker images -q rcland12/detection-stream:nginx-latest | xargs -I{} docker tag {} rcland12/detection-stream:nginx-${LATEST_VERSION}
docker push rcland12/detection-stream:jetson-${LATEST_VERSION}
docker push rcland12/detection-stream:jetson-triton-${LATEST_VERSION}
docker push rcland12/detection-stream:nginx-${LATEST_VERSION}
docker rmi -f rcland12/detection-stream:jetson-${LATEST_VERSION}
docker rmi -f rcland12/detection-stream:jetson-triton-${LATEST_VERSION}
docker rmi -f rcland12/detection-stream:nginx-${LATEST_VERSION}
docker push rcland12/detection-stream:jetson-latest
docker push rcland12/detection-stream:jetson-triton-latest
docker push rcland12/detection-stream:nginx-latest