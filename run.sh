#!/bin/bash
source .env

if [ "${OBJECT_DETECTION}" == "True" ]; then
    docker-compose up -d triton
    python3 app/main.py
elif [ "${OBJECT_DETECTION}" == "False" ]; then
    /bin/bash app/stream.sh
else
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
    exit 120
fi