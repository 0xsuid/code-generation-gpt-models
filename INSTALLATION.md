# Installation

## Using Automated BASH Script

- Install Nvidia Driver, CUDA Toolkit & Python Dependencies

Stable installation from requirements.txt

```bash
chmod +x install.sh stable
./install.sh
```

Install latest version of libraries:

```bash
chmod +x install.sh latest
./install.sh
```

## Nvidia Drivers - Optional

Optional - Remove previously installed nvidia driver

```bash
sudo apt clean
sudo apt update
sudo apt purge nvidia-* 
```

nvidia driver version 510 & CUDA 11.6 - [download](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt install nvidia-driver-510 cuda-toolkit-11-6
```

## Dependency Installation

```bash
pip install pandas transformers datasets accelerate nvidia-ml-py3 python-dotenv requests
```

```bash
pip install torch torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu116
```

```bash
sudo apt install libaio-dev
pip install deepspeed
```

If error "deepspeed: command not found" is visible after installation - faced on ubuntu 20:

```bash
nano ~/.bashrc
```

Save following at end of file

```bash
export PATH="/home/user/.local/bin:$PATH"
```

You will then need to profile, do this by either running the command:

```bash
source ~/.bashrc
```
