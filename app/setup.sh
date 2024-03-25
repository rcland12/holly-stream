#!/bin/bash

sudo apt update -y
sudo apt install -y libcamera-dev libcamera-apps python3-libcamera ffmpeg

python3 -m venv $PWD/.stream_env --system-site-packages
$PWD/.stream_env/bin/python3 -m pip install --upgrade pip
$PWD/.stream_env/bin/python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
$PWD/.stream_env/bin/python3 -m pip install opencv-python imutils python-dotenv tritonclient[all]
