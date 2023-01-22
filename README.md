# Code Generation with GPT-Neo Models

This project demonstrates the use of GPT-Neo models for code generation. GPT-Neo is a variant of the GPT (Generative Pre-trained Transformer) model, which is large scale autoregressive language model. In this project, we use GPT-Neo to generate code in Python programming language.

## Requirements

- Python 3.8 or higher
- Pytorch 1.13 or higher
- Hugging Face's transformers library

## Getting Started

Clone the repository

```bash
git clone https://github.com/0xsuid/code-generation-gpt-models.git
```

Install Nvidia Driver, CUDA Toolkit & Python Dependencies

```bash
chmod +x install.sh
./install.sh
```

Fine-Tune the model on apps dataset  
**Note**: argument "-u" is required to disable python output buffering

Python

```bash
nohup python3 -u tune_gpt.py --limit 10 > output.log 2>&1 &
```

Deepspeed

```bash
nohup deepspeed --num_gpus=4 tune_gpt.py --deepspeed deepspeed.json > output.log 2>&1 &
```

## Limitations

- The generated code may not always be syntactically correct or runnable.
- The model is only as good as the dataset it is trained on, so the quality of the generated code will depend on the diversity and quality of the training data.
GPT-Neo models are large, so they require a powerful GPU and a lot of memory to train.

## Conclusion

Code generation with GPT-Neo models is a promising approach for automating repetitive coding tasks. With the right dataset and fine-tuning, it can be used to generate high-quality code in a variety of programming languages. However, it still has some limitations, and it is not a substitute for human programmers.

## References
- [GPT Neo](https://github.com/EleutherAI/gpt-neo)
- [Measuring Coding Challenge Competence With APPS](https://arxiv.org/pdf/2105.09938.pdf)
