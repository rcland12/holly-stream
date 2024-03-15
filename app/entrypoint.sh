#!/bin/bash
if [ "${OBJECT_DETECTION}" == "True" ]; then
    python3 main.py
elif [ "${OBJECT_DETECTION}" == "False" ]; then
    /bin/bash stream.sh
else
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
    exit 120
fi