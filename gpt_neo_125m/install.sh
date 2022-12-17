#!/bin/bash

export DEBIAN_FRONTEND=noninteractive
sudo apt update -y
sudo apt upgrade -y

## Remove default Nvidia Driver
sudo apt clean -y
sudo apt update -y
sudo apt purge nvidia-* -y

## Install Nvidia driver 510 & Cuda toolkit 11.6
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb -P /tmp
sudo dpkg -i /tmp/cuda-keyring_1.0-1_all.deb
sudo apt-get update -y
sudo apt install -y nvidia-driver-510 cuda-toolkit-11-6

# Update bashrc - For Ubuntu20.0
# To resolve -> WARNING: The script datasets-cli is installed in '/home/user/.local/bin' which is not on PATH.
echo 'export PATH="/home/user/.local/bin:$PATH"' >> ~/.bashrc 
source ~/.bashrc

# Install Python dependencies
sudo apt-get install -y python3-pip
python3 -m pip install pandas transformers datasets accelerate nvidia-ml-py3 python-dotenv requests
python3 -m pip install torch torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu116

# Install Deepspeed
sudo apt install -y libaio-dev
python3 -m pip install deepspeed