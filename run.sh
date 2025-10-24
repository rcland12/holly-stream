#!/bin/bash

source .env

export PATH="${DOCKER_COMPOSE_PATH}:${PATH}"

if [ -z $OBJECT_DETECTION ]; then echo "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False."; fi

if [[ "${OBJECT_DETECTION}" == "True" ]]; then
    docker-compose up -d triton

    echo "Waiting to start Holly Stream until Triton is healthy."
    for ((attempt=1; attempt<=60; attempt++)); do
        if docker-compose exec triton curl -s -f "http://localhost:8000/v2/health/ready" > /dev/null; then
            break
        fi
        sleep 1
        [[ $attempt -eq 60 ]] && echo "Triton failed all health checks after 60 seconds. Stopping all services." && exit 120
    done
fi

docker-compose up -d app
echo "Holly Stream has started. Performing health check..."

for i in {1..12}; do
    if [ "$( docker container inspect -f '{{.State.Running}}' holly-stream-app )" = "true" ]; then
        echo "Holly Stream STATUS: HEALTHY"
        break
    elif [ $i -eq 12 ]; then
        echo "Holly STREAM STATUS: UNHEALTHY"
        echo "Shutting down."
        docker-compose down
        exit 1
    else
        echo "Health check attempt: $i/12"
        sleep 5
    fi
done

echo "Starting fan..."
echo 255 > /sys/devices/pwm-fan/target_pwm

echo "System running."
