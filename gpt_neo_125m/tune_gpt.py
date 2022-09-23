import json
import torch
import pandas as pd
from pynvml import *
# from urllib import parse
from datasets import load_dataset
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, logging

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def format_input(dataset):
        formatted_dataset = []
        for idx, data in enumerate(dataset):
                answer_type = "\nUse Call-Based format\n" if len(data["starter_code"])>0 else "\nUse Standard Input format\n"
                str_format = "\nQUESTION:\n" + data['question'] + "\n" + data["starter_code"] + "\n" + answer_type + "\nANSWER:\n"
                answers = json.loads(data["solutions"])
                for answer in answers:
                        formatted_dataset.append(str_format + answer)
        return formatted_dataset


# torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", 
                                            bos_token='<|startoftext|>',
                                            eos_token='<|endoftext|>', 
                                            pad_token='<|pad|>')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").cuda()
model.resize_token_embeddings(len(tokenizer))

raw_ds = load_dataset("codeparrot/apps", split="train")
coding_problems = format_input(raw_ds)
max_length = max([len(tokenizer.encode(coding_problem)) for coding_problem in coding_problems])
print("Max length: {}".format(max_length))


class AppsDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', 
                                        truncation=True,
                                        max_length=max_length, 
                                        padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


train_dataset = AppsDataset(descriptions, tokenizer, max_length=max_length)
save_dir = './results'

# Logging - https://huggingface.co/docs/transformers/main_classes/logging
# TRANSFORMERS_VERBOSITY=error ./tune_gpt.py
logging.set_verbosity_error()
# logging.set_verbosity_info()

default_args = {
    "output_dir": save_dir, 
    # overwrite_output_dir: False,
    
    # Disable Evaluation
    "evaluation_strategy": 'no',
    "do_eval": False,
    "eval_steps": 0, 
    
    # Logging
    "log_level": "error",
    "logging_first_step": True,
    "logging_steps": 5,
    "logging_dir": './logs',
    
    # Save
    "save_steps": 200,
    "save_total_limit": 2,
    
    # Total number of training epochs to perform
    "num_train_epochs": 5,
    "per_device_train_batch_size": 2,
    
    # The idea behind gradient accumulation is to instead of calculating the gradients for the whole batch at once to do it in smaller steps. 
    # The way we do that is to calculate the gradients iteratively in smaller batches by doing a forward and backward pass through the model and accumulating the gradients in the process. 
    # When enough gradients are accumulated we run the model’s optimization step. This way we can easily increase the overall batch size to numbers that would never fit into the GPU’s memory. 
    # In turn, however, the added forward and backward passes can slow down the training a bit.
    "gradient_accumulation_steps": 4,
    
    # In order to compute the gradients during the backward pass all activations from the forward pass are normally saved. 
    # This can create a big memory overhead. Alternatively, one could forget all activations during the forward pass and recompute them on demand during the backward pass. 
    # This would however add a significant computational overhead and slow down training.
    "gradient_checkpointing":True,
    
    # Drop the last incomplete batch if it is not divisible by the batch size
    "dataloader_drop_last": True,
    
    # Number of steps used for a linear warmup from 0 to learning_rate. Overrides any effect of warmup_ratio.
    "warmup_steps": 100, 
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.
    "weight_decay": 0.01, 
    # The initial learning rate for AdamW optimizer.
    "learning_rate": 5e-5,
    
    # we can reduce the precision the variales and their computations are faster. 
    "fp16": True
}
training_args = TrainingArguments(**default_args)
trainer = Trainer(model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[0] for f in data])}
        )

print_gpu_utilization()
result = trainer.train()
print_summary(result)

model.save_pretrained(os.path.join(save_dir, "final_checkpoint"))