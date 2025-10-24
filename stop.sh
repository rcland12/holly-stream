#!/bin/bash

source .env

export PATH="${DOCKER_COMPOSE_PATH}:${PATH}"

docker-compose down

echo 0 > /sys/devices/pwm-fan/target_pwm
