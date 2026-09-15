import argparse
import os
import re
import signal
import socket
import subprocess
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from main import (
    EnvArgumentParser,
    add_camera_args,
    build_camera_command,
    camera_settings,
    read_exactly,
)


def next_index(save_path: str, prefix: str) -> int:
    """
    Find the index after the highest image already saved with this prefix, so repeated runs add to a dataset.

    Args:
        save_path (str): Directory the images are saved to.
        prefix (str): Filename prefix, e.g. the camera's hostname.

    Returns:
        int: The index to start numbering at.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.jpg$")
    indexes = [
        int(match.group(1))
        for match in map(pattern.match, os.listdir(save_path))
        if match
    ]
    return max(indexes) + 1 if indexes else 0


def main(save_path: str, number_to_save: int, period: float, prefix: str) -> None:
    """
    Save a camera frame every `period` seconds for building a training dataset.

    Frames come from rpicam-vid with the same size, orientation and image tuning as main.py, so the
    training images look like what the model sees on the stream. The camera can only be opened by one
    process, so stop the app container first.

    Args:
        save_path (str): Directory to save images to, named <prefix>_00000.jpg, <prefix>_00001.jpg, ...
        number_to_save (int): Number of images to save.
        period (float): Seconds between saved images.
        prefix (str): Filename prefix.
    """
    load_dotenv()
    parser = EnvArgumentParser()
    add_camera_args(parser)
    settings = camera_settings(parser.parse_args())
    width, height = settings["camera_width"], settings["camera_height"]

    os.makedirs(save_path, exist_ok=True)
    index = next_index(save_path, prefix)
    last_index = index + number_to_save

    camera = subprocess.Popen(
        build_camera_command(**settings), stdout=subprocess.PIPE, bufsize=0
    )
    frame = bytearray(width * height * 3 // 2)

    # Give auto exposure and white balance a moment to settle before the first image
    next_save = time.monotonic() + 2
    try:
        while index < last_index:
            if not read_exactly(camera.stdout, frame):
                raise RuntimeError(f"rpicam-vid exited with code {camera.wait()}")
            if time.monotonic() < next_save:
                continue

            yuv = np.frombuffer(frame, dtype=np.uint8).reshape(height * 3 // 2, width)
            path = os.path.join(save_path, f"{prefix}_{index:05d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420), [cv2.IMWRITE_JPEG_QUALITY, 95])
            index += 1
            print(f"Saved {path} ({number_to_save - (last_index - index)}/{number_to_save})", flush=True)
            next_save += period

    finally:
        camera.terminate()
        try:
            camera.wait(timeout=10)
        except subprocess.TimeoutExpired:
            camera.kill()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: exit(0))

    parser = argparse.ArgumentParser(
        description="Save camera frames at an interval for training a custom model. Camera settings come from .env."
    )
    parser.add_argument("--output", default="data/images", help="Directory to save images to")
    parser.add_argument("--count", type=int, default=200, help="Number of images to save")
    parser.add_argument("--period", type=float, default=5, help="Seconds between images")
    parser.add_argument("--prefix", default=socket.gethostname(), help="Filename prefix (default: hostname)")
    args = parser.parse_args()

    main(args.output, args.count, args.period, args.prefix)
