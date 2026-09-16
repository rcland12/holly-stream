#!/bin/bash
#
# Builds and pushes the Linux camera and detector images as :linux-camera-latest / :linux-detector-latest and
# :linux-camera-<LATEST_VERSION> / :linux-detector-<LATEST_VERSION>.
# Reads DOCKER_USERNAME, DOCKER_PASSWORD and LATEST_VERSION from .env in the repo root.

set -euo pipefail

cd "$(dirname "$0")"
source .env

for var in DOCKER_USERNAME DOCKER_PASSWORD LATEST_VERSION; do
    if [[ -z "${!var:-}" ]]; then
        echo "Set ${var} in your .env file"
        exit 1
    fi
done

echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin

REPO="rcland12/detection-stream"
docker build -t "${REPO}:linux-camera-latest" camera
docker compose -f server/compose.yml --profile detection build holly-detector

for image in linux-camera linux-detector; do
    docker tag "${REPO}:${image}-latest" "${REPO}:${image}-${LATEST_VERSION}"
    docker push "${REPO}:${image}-${LATEST_VERSION}"
    docker push "${REPO}:${image}-latest"
done

echo "Finished pushing ${REPO}:linux-{camera,detector}-{${LATEST_VERSION},latest}."
