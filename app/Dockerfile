FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /root/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        alsa-utils \
        ca-certificates \
        curl \
        ffmpeg \
        libasound2 \
        libasound2-plugins \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        v4l-utils && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
        imutils \
        opencv-python-headless \
        python-dotenv \
        "tritonclient[all]"

COPY main.py stream.sh entrypoint.sh ./
COPY ./images/ ./images/
RUN chmod +x entrypoint.sh stream.sh

CMD ["/bin/bash", "entrypoint.sh"]
