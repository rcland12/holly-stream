#!/bin/bash

sudo apt update -y
sudo apt install -y libcamera-dev libcamera-apps python3-libcamera

python3 -m venv ~/.stream_env --system-site-packages
source ~/.stream_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install opencv-python imutils python-dotenv tritonclient[all]