#!/bin/bash

source .env

if [ -z $OBJECT_DETECTION ]; then echo "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False."; fi

if [ "${OBJECT_DETECTION}" == "True" ]; then
    docker compose down triton app

elif [ "${OBJECT_DETECTION}" == "False" ]; then
    docker compose down app

else
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
    exit 120
fi
