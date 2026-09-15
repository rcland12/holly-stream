#!/bin/bash

cd "$(dirname "$0")"
source .env

export PATH="${DOCKER_COMPOSE_PATH}:${PATH}"

docker-compose down
