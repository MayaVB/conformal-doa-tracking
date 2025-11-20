# Docker Setup for Conformal DOA Tracking

This directory contains Docker configurations to recreate the development environment for both UNet training and dataset generation.

## Prerequisites

- Docker with NVIDIA Container Runtime
- NVIDIA GPU drivers
- Docker Compose (optional, for easier management)

## Images

### UNet Training Environment (`docker/unet/`)
- Based on NVIDIA PyTorch 22.11 container
- Configured for U-Net model training and testing
- Includes ML packages like PyTorch, wandb, scikit-learn

### Dataset Generation Environment (`docker/dataset_gen/`)
- Based on NVIDIA PyTorch 22.11 container  
- Configured for acoustic data generation
- Includes audio processing packages like librosa, pyroomacoustics, gpuRIR

## Building Images

### Option 1: Using Docker Compose (Recommended)

```bash
# Build both images
docker-compose build

# Run UNet training environment
docker-compose run unet

# Run dataset generation environment
docker-compose run dataset_gen
```

### Option 2: Using Docker directly

```bash
# Build UNet image
docker build -f docker/unet/Dockerfile -t conformal-doa/unet:latest .

# Build dataset generation image
docker build -f docker/dataset_gen/Dockerfile -t conformal-doa/dataset_gen:latest .

# Run UNet container
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace/unet \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -it conformal-doa/unet:latest

# Run dataset generation container
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace/unet \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -it conformal-doa/dataset_gen:latest
```

## Wandb Setup

After starting a container, you'll need to login to wandb:

```bash
wandb login
```

Enter your API key when prompted. This keeps your credentials out of the container image.

## Customization

- Modify `requirements.txt` files in each docker directory to add/remove Python packages
- Update Dockerfiles to change base images or add system packages
- Adjust docker-compose.yml for different volume mounts or environment variables