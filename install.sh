#!/bin/bash

INSTALLATION_TYPE=${1:-stable}

export DEBIAN_FRONTEND=noninteractive
sudo apt update -y
sudo apt upgrade -y

# Install Nvidia driver 510 & Cuda toolkit 11.6
# Install cudakeyring if it is missing
NVIDIA_KEYRING=$(apt-key list 2> /dev/null | grep cudatools@nvidia.com)
if [[ ! $NVIDIA_KEYRING ]]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb -P /tmp
    sudo dpkg -i /tmp/cuda-keyring_1.0-1_all.deb
    sudo apt-get update -y
fi

# Check Nvidia driver version for cuda 11.6 compatibility
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id3
NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0)
NVIDIA_DRIVER_MAJOR_VERSION=${NVIDIA_DRIVER_VERSION%%.*}
if [[  $NVIDIA_DRIVER_MAJOR_VERSION -gt 450 ]]; then
    sudo apt install -y cuda-toolkit-11-6
else
    # Remove default Nvidia Driver
    sudo apt clean -y
    sudo apt purge nvidia-* -y
    sudo apt install -y nvidia-driver-510 cuda-toolkit-11-6
fi

# Install Git LFS - To upload large files on HuggingFace
sudo apt install git git-lfs -y
git lfs install

# For Deepspeed
sudo apt install -y libaio-dev ninja-build python3-dev

# Update bashrc - For Ubuntu20.0
# To resolve -> WARNING: The script datasets-cli is installed in '/home/user/.local/bin' which is not on PATH.
echo 'export PATH="/home/user/.local/bin:$PATH"' >> ~/.bashrc 
source ~/.bashrc

# Install Python dependencies
sudo apt-get install -y python3-pip

if [ "$INSTALLATION_TYPE" == "stable" ]; then
    python3 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
elif [ "$INSTALLATION_TYPE" == "latest" ]; then
    python3 -m pip install transformers tensorboard datasets nvidia-ml-py3 python-dotenv requests huggingface_hub evaluate pyext accelerate
    python3 -m pip install torch --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu116

    # Install Deepspeed
    python3 -m pip install deepspeed
fi

# Reboot server - to update nvidia driver
sudo reboot now
