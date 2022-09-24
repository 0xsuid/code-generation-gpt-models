# Code generation with GPT NEO 125M

Fine-Tuning GPT NEO 125M for code generation

## Execution

### Single GPU

```bash
nohup python3 tune_gpt.py --limit 10 > output.log 2>&1 &
```

## Recommendation

```bash
pip install pandas transformers datasets accelerate nvidia-ml-py3
```

## References

- https://huggingface.co/docs/transformers/perf_train_gpu_one
