# Contributing to Holly Stream

Thank you for your interest in contributing to Holly Stream! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Branch Strategy](#branch-strategy)
- [Development Environment Setup](#development-environment-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

Be respectful, professional, and constructive in all interactions. We aim to maintain a welcoming environment for all contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/holly-stream.git
   cd holly-stream
   ```
3. **Add the upstream repository** as a remote:
   ```bash
   git remote add upstream https://github.com/rcland12/holly-stream.git
   ```
4. **Switch to the appropriate branch** for your target platform (jetson, linux, or raspbian)

## Branch Strategy

This project uses a multi-branch strategy to support different hardware platforms:

- **master**: Contains project overview, license, and general documentation only
- **jetson**: NVIDIA Jetson platform implementation (Jetson Nano, JetPack OS)
- **linux**: Linux x86_64 implementation with NVIDIA GPU support
- **raspbian**: Raspberry Pi implementation (Pi 4, Debian-based OS)

### Contributing to a Specific Branch

When contributing, target the appropriate platform branch:

```bash
# For Jetson development
git checkout jetson

# For Linux development
git checkout linux

# For Raspberry Pi development
git checkout raspbian
```

Create your feature branch from the platform branch you're working on:

```bash
git checkout -b feature/your-feature-name
```

## Development Environment Setup

### Prerequisites

- Docker and Docker Compose
- Git
- Python 3.6+ (for non-Docker development)
- NVIDIA GPU with appropriate drivers (for GPU-accelerated inference)

### Platform-Specific Setup

#### Jetson
- NVIDIA Jetson device (tested on Jetson Nano)
- JetPack SDK 4.6.4+
- CSI or USB camera
- 4GB swap memory (recommended)

#### Linux
- Ubuntu 20.04+ or compatible distribution
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- USB or CSI camera

#### Raspberry Pi
- Raspberry Pi 4 (minimum)
- Raspbian 64-bit (Debian Bookworm or later)
- CSI or USB camera

### Environment Configuration

Create a `.env` file in the project root with your development settings:

```bash
OBJECT_DETECTION=True
MODEL=yolov5s
CONFIDENCE_THRESHOLD=0.3
IOU_THRESHOLD=0.25
STREAM_IP=127.0.0.1
STREAM_PORT=1935
STREAM_APPLICATION=live
STREAM_KEY=stream
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=22
```

## How to Contribute

### Reporting Bugs

1. Check the [Issues](https://github.com/rcland12/holly-stream/issues) page to see if the bug has already been reported
2. If not, create a new issue with:
   - Clear, descriptive title
   - Detailed description of the problem
   - Steps to reproduce
   - Expected vs. actual behavior
   - Platform/branch information
   - Hardware specifications
   - Relevant logs or error messages

### Suggesting Enhancements

1. Open an issue with the `enhancement` label
2. Provide a clear description of the proposed feature
3. Explain the use case and benefits
4. Include any implementation ideas or references

### Submitting Code Changes

1. **Keep changes focused**: One feature or fix per pull request
2. **Follow existing code style**: Match the patterns in the codebase
3. **Write clear commit messages**: Use descriptive, imperative messages
4. **Test thoroughly**: Ensure your changes work on the target platform
5. **Update documentation**: Modify README or other docs as needed

## Pull Request Process

1. **Update your fork** with the latest upstream changes:
   ```bash
   git fetch upstream
   git checkout jetson  # or linux/raspbian
   git merge upstream/jetson
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/descriptive-name
   ```

3. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "Add descriptive commit message"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/descriptive-name
   ```

5. **Open a Pull Request** on GitHub:
   - Select the appropriate base branch (jetson, linux, or raspbian)
   - Provide a clear title and description
   - Reference any related issues
   - Describe what was changed and why
   - Include testing details

6. **Respond to review feedback** promptly and make requested changes

7. **Squash commits** if requested before merging

### Pull Request Guidelines

- Target the correct platform branch (not master)
- Ensure Docker builds succeed
- Test the full streaming pipeline end-to-end
- Update README if adding features or changing configuration
- Add comments for complex logic
- Avoid unnecessary dependencies

## Coding Standards

### Python

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

### Example:

```python
def process_frame(frame: np.ndarray, confidence: float = 0.3) -> np.ndarray:
    """
    Process a video frame with object detection.

    Args:
        frame: Input frame as numpy array
        confidence: Detection confidence threshold

    Returns:
        Processed frame with bounding boxes
    """
    # Implementation
    pass
```

### Docker

- Keep Dockerfiles clean and well-commented
- Minimize layer count
- Use multi-stage builds where appropriate
- Pin dependency versions for reproducibility

### Configuration

- Use environment variables for configuration
- Provide sensible defaults
- Document all configuration options
- Validate input parameters

## Testing

### Before Submitting

1. **Build test**: Ensure Docker images build successfully
   ```bash
   docker compose build
   ```

2. **Run test**: Start the full application stack
   ```bash
   ./run.sh
   ```

3. **Stream test**: Verify streaming works on target platform
   - Test RTMP streaming to media player
   - Test HLS streaming in web browser
   - Test with object detection enabled and disabled

4. **Performance test**: Check resource usage and FPS
   - Monitor CPU/GPU utilization
   - Verify acceptable latency
   - Test with different model configurations

### Hardware Testing

Since this project targets specific hardware platforms, contributors should test on the relevant architecture when possible. If you don't have access to the target hardware, clearly note this in your pull request.

## Areas for Contribution

We welcome contributions in the following areas:

### Platform Support

- Support for additional NVIDIA Jetson models (Xavier, Orin, etc.)
- Support for AMD GPUs
- Support for Intel integrated graphics
- ARM64 optimization
- macOS support

### Streaming Enhancements

- WebRTC implementation for ultra-low latency
- DASH (Dynamic Adaptive Streaming over HTTP) support
- UDP streaming optimization
- Multi-bitrate streaming (adaptive quality)
- Audio support

### Object Detection Features

- YOLOv8/YOLOv9/v11 model support
- Other detection frameworks (SSD, Faster R-CNN, etc.)
- Object tracking (DeepSORT, ByteTrack)
- Pose estimation integration
- Segmentation support
- Custom class training workflows

### Performance Optimization

- Multi-camera support
- Frame buffering and queue management
- Model quantization improvements
- Batch processing optimization
- Memory usage reduction

### Deployment

- Kubernetes deployment manifests
- Ansible playbooks for automated setup
- Systemd service files
- Docker Swarm configuration
- Auto-restart and health monitoring

### Documentation

- Tutorial videos or blog posts
- Architecture diagrams
- Troubleshooting guides
- Performance benchmarking
- Use case examples

### Code Quality

- Unit tests and integration tests
- CI/CD pipeline setup
- Code linting and formatting
- Security vulnerability scanning
- Dependency updates

### User Experience

- Web UI for configuration
- Mobile app for viewing streams
- CLI improvements
- Configuration wizard
- Stream preview without full setup

## Development Tips

### Debugging

Enable verbose logging in your `.env`:
```bash
LOG_LEVEL=DEBUG
```

Access container logs:
```bash
docker compose logs -f app
docker compose logs -f triton
```

### Testing Models Locally

Use the provided test script:
```bash
cd triton
python test.py
```

### Converting Models

Convert ONNX models to TensorRT:
```bash
cd triton
./convert.sh path/to/model.onnx
```

## Questions?

If you have questions about contributing:

1. Check existing issues and pull requests
2. Review the branch-specific README
3. Open a discussion issue on GitHub
4. Reach out through GitHub Issues

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE) file).

---

Thank you for contributing to Holly Stream! Your efforts help make this project better for everyone.
