# Code Generation with GPT-Neo Models

This project demonstrates the use of GPT-Neo models for code generation. GPT-Neo is a variant of the GPT (Generative Pre-trained Transformer) model, which is large scale autoregressive language model. In this project, we use GPT-Neo to generate code in Python programming language.

## Requirements

- Python 3.8 or higher
- Pytorch 1.13 or higher
- Hugging Face's transformers library
- GPU

## Getting Started

Clone the repository

```bash
git clone https://github.com/0xsuid/code-generation-gpt-models.git
```

Install Nvidia Driver, CUDA Toolkit & Python Dependencies
For More info check INSTALLATION.md

```bash
chmod +x install.sh
./install.sh
```

Fine-Tune the model on apps dataset  

### Single GPU

**Note**: argument "-u" is required to disable python output buffering

```bash
nohup python3 -u tune_gpt.py --limit 10 --local-rank 0 > output.log 2>&1 &
```

### Single GPU/MultiGPU with Deepspeed

```bash
nohup deepspeed tune_gpt.py --deepspeed deepspeed.json > output.log 2>&1 &
```

### Supported Arguments

1. Limit
    - "--limit" - "Limit Total no. of problems"
2. Upload Model
    - "--upload-model" - "Upload fine-tuned model to Huggingface"
3. Stop
    - "--stop-instance" - "Stop tensordock instance after training"
4. Local Rank
    - "--local-rank" - "Local rank for deepspeed, it should be 0 when not using deepspeed to save model"
5. Upload Experiement
    - "--upload-experiment"" - "Upload Experiment directory to huggingface repo"
6. Verbosity
    - "--verbosity"

## Logs Visualization with Tensorboard

```bash
tensorboard --logdir experiments/2022-10-15-9e416bbdeafeaea88e8747a0edd284f93d7551ea3cc387377269ceed52957730/logs
```

## Labels

we pass the input data as the label instead of just the answer labels. This is because we are training a language model, hence we want the model to learn the pattern of the prompt and not just answer class. In a sense, the model learns to predict the words of the input question + answer structured in the prompt, and in the process learn the code generation task.

- https://huggingface.co/docs/transformers/v4.23.1/en/glossary#labels
- https://huggingface.co/transformers/v4.8.2/model_doc/gpt_neo.html#transformers.GPTNeoForCausalLM.forward

## Cuda out of memory

When Using Multi-GPU Environment and first gpu run out of memory but we have more memory available on other gpus then setting "max_split_size_mb" might be useful

- https://stackoverflow.com/questions/73747731/runtimeerror-cuda-out-of-memory-how-setting-max-split-size-mb

## Decoding Strategies

- https://huggingface.co/docs/transformers/main/en/generation_strategies#decoding-strategies

## Limitations

- The generated code may not always be syntactically correct or runnable.
- The model is only as good as the dataset it is trained on, so the quality of the generated code will depend on the diversity and quality of the training data.
GPT-Neo models are large, so they require a powerful GPU and a lot of memory to train.

## Conclusion

Code generation with GPT-Neo models is a promising approach for automating repetitive coding tasks. With the right dataset and fine-tuning, it can be used to generate high-quality code in a variety of programming languages. However, it still has some limitations, and it is not a substitute for human programmers.


## References

- [GPT Neo](https://github.com/EleutherAI/gpt-neo)
- [Measuring Coding Challenge Competence With APPS](https://arxiv.org/pdf/2105.09938.pdf)
- https://huggingface.co/docs/transformers/perf_train_gpu_one
- https://huggingface.co/docs/transformers/main/en/generation_strategies#decoding-strategies
