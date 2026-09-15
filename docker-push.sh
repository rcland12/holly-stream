#!/bin/bash

source .env

for var in DOCKER_USERNAME DOCKER_PASSWORD LATEST_VERSION; do
    if [[ -z "${!var}" ]]; then
        echo "Set ${var} in your .env file"
        exit 1
    fi
done

echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin

REPO="rcland12/detection-stream"
IMAGES=(
    raspbian
    raspbian-triton
    nginx
)

for IMAGE in "${IMAGES[@]}"; do
    SRC_TAG="${REPO}:${IMAGE}-latest"
    DST_TAG="${REPO}:${IMAGE}-${LATEST_VERSION}"
    IMAGE_ID="$(docker images -q "${SRC_TAG}" || true)"

    if [[ -z "${IMAGE_ID}" ]]; then
        echo "WARNING: Source image not found locally: ${SRC_TAG} (skipping)"
        continue
    fi

    docker tag "${IMAGE_ID}" "${DST_TAG}"
    docker push "${DST_TAG}"
    docker rmi -f "${DST_TAG}" || true
    docker push "${SRC_TAG}"
done

echo "Finished pushing TAG=${LATEST_VERSION} and TAG=latest images."