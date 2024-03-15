#!/bin/bash
if [ "${OBEJCT_DETECTION}" == "True" ]; then
    python3 main.py
elif [ "${OBJECT_DETECTION}" == "False" ]; then
    /bin/bash stream.sh
else
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
fi