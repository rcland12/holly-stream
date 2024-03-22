FROM debian:bullseye-slim
WORKDIR /root/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 python3-pip python3-dev ffmpeg build-essential && \
    pip3 install --upgrade --no-cache-dir pip && \
    python3 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir opencv-python \
    imutils python-dotenv tritonclient[all]

RUN wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh && \
    chmod +x install_pivariety_pkgs.sh && \
    ./install_pivariety_pkgs.sh -p libcamera_dev && \
    ./install_pivariety_pkgs.sh -p libcamera_apps && \
    rm install_pivariety_pkgs.sh

COPY main.py utilities.py stream.sh entrypoint.sh ./
CMD ["/bin/bash", "entrypoint.sh"]