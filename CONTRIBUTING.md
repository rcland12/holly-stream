# Contributing to Holly Stream

Thank you for your interest in contributing. This guide explains how the repository is organized, how to work on
each platform, and what a pull request needs.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How the repository is organized](#how-the-repository-is-organized)
- [Getting started](#getting-started)
- [Working on each branch](#working-on-each-branch)
- [Things shared across branches](#things-shared-across-branches)
- [Reporting bugs and suggesting features](#reporting-bugs-and-suggesting-features)
- [Pull requests](#pull-requests)
- [Coding standards](#coding-standards)
- [Testing](#testing)
- [Areas for contribution](#areas-for-contribution)

## Code of Conduct

Be respectful, professional and constructive in all interactions.

## How the repository is organized

Each hardware platform is a separate branch with its own pipeline. The platform branches are never merged into each
other.

| Branch | What it contains |
|---|---|
| `master` | Project overview, license and this guide. No code. |
| `raspbian` | Raspberry Pi camera (`camera/`: GStreamer + systemd) and the Holly server (`server/`: MediaMTX ingest + GPU detector) |
| `linux` | Linux USB webcam camera (`camera/`: FFmpeg in Docker) and the Holly server (`server/`) |
| `jetson` | Jetson Nano: DeepStream pipeline in C (`app/`), models and training scripts (`models/`), nginx-rtmp web player (`nginx/`) |
| `raspbian_develop`, `linux_develop`, `jetson_develop` | Work in progress for each platform |

Changes land on a `*_develop` branch first and reach the platform branch through a pull request once they have been
tested on the hardware. Users clone the platform branches, so those must always work.

## Getting started

1. **Fork** the repository on GitHub.
2. **Clone your fork** with the branch you will work on. Clone each platform into its own directory if you work on
   more than one:
   ```bash
   git clone -b linux_develop https://github.com/YOUR-USERNAME/holly-stream.git holly-stream
   git clone -b raspbian_develop https://github.com/YOUR-USERNAME/holly-stream.git holly-stream-raspbian
   git clone -b jetson_develop https://github.com/YOUR-USERNAME/holly-stream.git holly-stream-jetson
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/rcland12/holly-stream.git
   ```
4. **Create a feature branch** from the develop branch:
   ```bash
   git checkout -b feature/short-description
   ```

## Working on each branch

Each branch's README covers setup, configuration and troubleshooting in full. What follows is what you need to
develop on it.

### `raspbian`

- **Camera** (`camera/holly-camera.sh`, run by the `holly-camera` systemd service): needs a Raspberry Pi 4 or 5 on
  Raspberry Pi OS Bookworm or Trixie with a camera module. After editing, reinstall with `sudo ./camera/install.sh`
  and watch `journalctl -u holly-camera -f`. GStreamer before 1.24 (Bookworm) behaves differently from Trixie, so
  say which one you tested.
- **Server** (`server/`): needs Linux with Docker, and an NVIDIA GPU (Turing or newer, driver 560+, NVIDIA Container
  Toolkit) for the detector. Rebuild and run with
  `docker compose --profile detection up -d --build` and follow `docker compose logs -f holly-detector`.
  `./server/status.sh` shows what each camera is doing. Set `LOG_STATS=true` to log fps and timings.

### `linux`

- **Camera** (`camera/`): needs Linux with Docker and a UVC webcam. Rebuild with
  `docker compose -f camera/compose.yml --profile '*' build`, then `./stop.sh && ./run.sh` and
  `docker logs -f holly-camera`. The encoder path depends on the hardware (NVENC, VAAPI or x264), so test the ones
  you touch; `ENCODER=x264` forces the CPU path on any machine.
- **Server** (`server/`): as for `raspbian`, above.

A camera and a server can run on the same machine (`SERVER_HOST=127.0.0.1`), which is the quickest way to test the
whole chain.

### `jetson`

- Needs a Jetson Nano 4GB with JetPack 4.6.x (DeepStream 6.0.1, TensorRT 8.2), a CSI camera, and Docker with the
  NVIDIA runtime as the default.
- The pipeline is `app/src/holly_stream.c` with the YOLO output parser in `app/src/nvdsparsebbox_yolo.cpp`, built
  by `app/src/Makefile` inside the Docker image. Build on the Jetson with `docker-compose build`, then `./run.sh`.
- `docker logs -f holly-stream-app | grep STATS` shows capture, inference and stream fps, latency and upload rate.
- Models must be exported with `models/export.py` (TensorRT 8.2-compatible ops, `[N, 6]` output). A new model
  builds its engine on first start, which takes 10-15 minutes on a Nano.

## Things shared across branches

Some files and interfaces are used by more than one branch. Keep them compatible, and make the same change on every
branch that has them.

- **`all-cameras.sh`** (and `run-all-cameras.sh`, `stop-all-cameras.sh`, `status-all-cameras.sh`) is identical on
  every branch.
- **`run.sh`, `stop.sh` and `status.sh`** at the top of every branch are what the multi-camera scripts call over
  SSH. They must:
  - run without prompts, over SSH with no terminal;
  - end with a one-line result such as `<hostname>: started`, `<hostname>: stopped` or `<hostname>: streaming ...`;
  - exit non-zero on failure.
- **Never start on boot.** Every camera runs only between `run.sh` and `stop.sh`, restarting itself after errors
  in between. Keep that behavior: Docker containers check `HOLLY_BOOT_ID`, and the Pi's systemd unit has no
  `[Install]` section.
- **`server/`** is the same Holly server in `raspbian` and `linux`. A detector or ingest change on one belongs on the
  other.
- **The camera protocol**: cameras publish MPEG-TS over SRT to the ingest with the stream id
  `publish:<stream name>:::<options>`. The detector reads `detect=1` and `rotate=180` from the options. A new option
  must be ignored safely by older detectors and cameras.
- **Model format**: the server detector expects ONNX with one `[1, 3, H, W]` input and one `[1, N, 6]` output of
  `x1, y1, x2, y2, score, class`, and the class names in the Ultralytics metadata.

## Reporting bugs and suggesting features

Check [GitHub Issues](https://github.com/rcland12/holly-stream/issues) first. A new bug report should include:

- the branch and commit;
- the hardware: device, GPU and driver, camera, OS version;
- what you did, what you expected and what happened;
- logs: `journalctl -u holly-camera` (Pi), `docker logs holly-camera` (Linux camera),
  `docker compose logs holly-ingest holly-detector` (server), `docker logs holly-stream-app` (Jetson), and the
  output of `./status.sh` or `./server/status.sh`.

For a feature, describe the use case, the platform or platforms it affects, and any implementation ideas.

## Pull requests

1. **Keep your branch current**:
   ```bash
   git fetch upstream
   git rebase upstream/linux_develop   # or raspbian_develop / jetson_develop
   ```
2. **Keep each pull request focused**: one fix or feature, on one platform unless it touches shared files.
3. **Open the pull request against the matching `*_develop` branch**, never against `master` or a platform branch
   directly (documentation changes to this guide or the overview go to `master`).
4. **Describe**:
   - what changed and why;
   - the hardware you tested on and how (see [Testing](#testing));
   - before and after numbers for performance changes (fps, inference time, CPU and GPU use, latency).
5. **Update the documentation**: the branch README for anything a user sees, and `.env.example` or
   `camera.env.example` for every new setting, with its default and a short explanation.

## Coding standards

Match the style of the file you are editing. In general:

### Shell

- `#!/bin/bash` and `set -euo pipefail` for scripts that do work; quote every variable expansion.
- A header comment saying what the script does, how it is run and what it prints.
- Validate settings early and fail with a message that names the setting and the value it got.
- Run [ShellCheck](https://www.shellcheck.net/) on scripts you change.

### Python (server detector)

- Python 3.12, type hints, and docstrings for classes and non-obvious functions.
- Keep per-frame work on the GPU where it already is; a copy between the CPU and the GPU on every frame is a
  performance regression.
- Settings come from environment variables in `holly/config.py`, with defaults that work.
- Pin dependency versions in `requirements.txt`.

### C / C++ (Jetson)

- C for the pipeline, C++11 for the nvinfer parser, building cleanly with `-Wall` against DeepStream 6.0.1 headers.
- Target JetPack 4.6 / TensorRT 8.2; newer APIs are not available on the Nano.

### Docker and configuration

- Pin base images and package versions. Comment anything a reader would not guess (runtime capabilities, device
  rules, why a layer is separate).
- Every setting gets a documented entry in the example env file and the README configuration table.
- Defaults should work for the common case without editing.

## Testing

There is no automated test suite; the pipelines depend on cameras and GPUs, so pull requests are tested on
hardware. Before submitting, check:

1. **Build**: the Docker images build (`docker compose build`, or `docker-compose build` on the Jetson), or the
   camera installs cleanly (`sudo ./camera/install.sh` on a Pi).
2. **Start and stop**: `./run.sh` reports `started`, `./status.sh` reports streaming, and `./stop.sh` stops
   it. Run them over SSH as well if you changed them.
3. **The stream**: it plays in a browser (HLS or WebRTC) and VLC, with audio if the camera has a microphone, both
   with detection on and off.
4. **Recovery**: the pipeline comes back on its own after the thing you changed fails. For example, restart the
   ingest, stop the detector, or disconnect the camera or network.
5. **Performance**: fps and inference time (`./server/status.sh`, `LOG_STATS=true`, or `STATS` lines on the
   Jetson), and CPU and GPU use, compared with before your change.
6. **Stability**: let it run for at least 30 minutes for anything touching timestamps, audio, encoding or
   networking; some problems only show up after the first few minutes.

If you cannot test on a platform your change affects, say so in the pull request.

## Areas for contribution

### Platforms

- Newer Jetsons (Orin Nano, Xavier) on current JetPack and DeepStream releases
- Jetson cameras publishing to the Holly server over SRT, like the Pi and Linux cameras
- A detector for non-NVIDIA GPUs or accelerators (Intel, AMD, Hailo, Coral)
- Testing the Linux camera on Intel and AMD GPUs (VAAPI) and on ARM boards

### Detection

- Object tracking on the server (e.g. ByteTrack), with IDs that persist across frames
- Batching several cameras into one TensorRT inference
- Segmentation and pose models
- Detection events: snapshots, webhooks or notifications when a class appears

### Streaming

- Adaptive bitrate for weak WiFi links
- Recording and clip export directly from the ingest
- Better WebRTC support, including audio

### Operations

- CI that builds the Docker images and runs ShellCheck
- Unit tests for the detector's pure-Python parts (box smoothing, overlays, config parsing)
- A metrics endpoint (e.g. Prometheus) for fps, latency and GPU use
- Documentation: setup walkthroughs, architecture notes, troubleshooting reports from your hardware

## License

By contributing, you agree that your contributions are licensed under the project's [MIT License](LICENSE).
