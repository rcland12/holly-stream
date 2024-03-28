#!/bin/bash

source .env

if [ -z $OBJECT_DETECTION ]; then echo "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False."; fi

if [ "${OBJECT_DETECTION}" == "True" ]; then
    docker compose up -d triton

    ATTEMPT=1
    RETRIES=60
    INTERVAL=1
    TOTAL_TIME=$((RETRIES * INTERVAL))

    echo "Waiting to start Holly Stream until Triton is healthy."
    while [ $ATTEMPT -le $RETRIES ]; do
        url="http://localhost:8000/v2/health/ready"
        response=$(curl --write-out "%{http_code}" --silent --output /dev/null "$url")

        if [ $response -eq 200 ]; then
            break
        else
            ATTEMPT=$((ATTEMPT + 1))
            sleep $INTERVAL
        fi
    done

    if [ $ATTEMPT -gt $RETRIES ]; then
        echo "Triton failed all health checks after $TOTAL_TIME. Stopping all services."
        exit 120
    fi

    docker compose up -d app

    echo "Holly Stream has started. Performing health check..."
    sleep 10

    if [ "$( docker container inspect -f '{{.State.Running}}' holly-stream-app-1 )" = "true" ]; then
        echo "Holly Stream STATUS: HEALTHY"
    else
        echo "Holly STREAM STATUS: UNHEALTHY"
        echo "Shutting down."
        docker compose down triton
    fi

elif [ "${OBJECT_DETECTION}" == "False" ]; then
    docker compose up -d app
    echo "Holly Stream has started. Performing health check..."
    sleep 10

    if [ "$( docker container inspect -f '{{.State.Running}}' holly-stream-app-1 )" = "true" ]; then
        echo "Holly Stream STATUS: HEALTHY"
    else
        echo "Holly STREAM STATUS: UNHEALTHY"
        echo "Shutting down."
    fi

else
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
    exit 120
fi
