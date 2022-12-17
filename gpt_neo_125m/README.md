# Code generation with GPT NEO 125M

Fine-Tuning GPT NEO 125M for code generation

## Execution

### Single GPU

Note: argument "-u" is required to disable python output buffering

```bash
nohup python3 -u tune_gpt.py --limit 10 > output.log 2>&1 &
```

```bash
nohup deepspeed --num_gpus=1 tune_gpt.py  --deepspeed deepspeed.json --upload-model > output.log 2>&1 &
```

## Supported Arguments

1. Limit
    - "--limit" - "Limit Total no. of problems"
2. Upload Model
    - "--upload-model" - "Upload fine-tuned model to Huggingface"
3. Stop
    - "--stop-instance" - "Stop tensordock instance after training"
4. Verbosity
    - "--verbosity"

## Logs Visualization with Tensorboard

```bash
tensorboard --logdir experiments/2022-10-15-9e416bbdeafeaea88e8747a0edd284f93d7551ea3cc387377269ceed52957730/logs
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

## Labels

we pass the input data as the label instead of just the answer labels. This is because we are training a language model, hence we want the model to learn the pattern of the prompt and not just answer class. In a sense, the model learns to predict the words of the input question + answer structured in the prompt, and in the process learn the code generation task.

- https://huggingface.co/docs/transformers/v4.23.1/en/glossary#labels
- https://huggingface.co/transformers/v4.8.2/model_doc/gpt_neo.html#transformers.GPTNeoForCausalLM.forward

## References

- https://huggingface.co/docs/transformers/perf_train_gpu_one
