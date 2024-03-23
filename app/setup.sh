#!/bin/bash

sudo apt-get update -y
# sudo apt-get install -y 

python3 -m venv ~/.stream_env
source ~/.stream_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install opencv-python imutils python-dotenv tritonclient[all]