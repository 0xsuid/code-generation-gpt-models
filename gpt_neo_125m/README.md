# Code generation with GPT NEO 125M

Fine-Tuning GPT NEO 125M for code generation

## Execution

### Single GPU

Note: argument "-u" is required to disable python output buffering

```bash
nohup python3 -u tune_gpt.py --limit 10 > output.log 2>&1 &
```

## Supported Arguments:

1. Limit
    - "--limit" - "Limit Total no. of problems"
2. Upload Model
    - "--upload-model" - "Upload fine-tuned model to Huggingface"
3. Stop
    - "--stop-instance" - "Stop tensordock instance after training"
4. Verbosity
    - "--verbosity"
## Logs Visualization with Tensorboard

```
tensorboard --logdir experiments/2022-10-15-9e416bbdeafeaea88e8747a0edd284f93d7551ea3cc387377269ceed52957730/logs
```

## Recommendation

```bash
pip install pandas transformers datasets accelerate nvidia-ml-py3 python-dotenv requests
```

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Nvidia Drivers - Optional

Optional - Remove previously installed nvidia driver

```bash
sudo apt clean
sudo apt update
sudo apt purge nvidia-* 
sudo apt autoremove
```

nvidia driver version 510 will provide -> CUDA 11.6

```bash
sudo apt install nvidia-driver-510 
```

After driver installation Force reinstall  pytorch

```bash
pip3 install torch torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu116
```

## References

- https://huggingface.co/docs/transformers/perf_train_gpu_one
