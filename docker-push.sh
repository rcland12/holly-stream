#!/bin/bash
#
# Builds and pushes the detector image as :detector-latest and :detector-<LATEST_VERSION>.
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
docker compose -f server/compose.yml build holly-detector

docker tag "${REPO}:detector-latest" "${REPO}:detector-${LATEST_VERSION}"
docker push "${REPO}:detector-${LATEST_VERSION}"
docker push "${REPO}:detector-latest"

echo "Finished pushing ${REPO}:detector-${LATEST_VERSION} and ${REPO}:detector-latest."
