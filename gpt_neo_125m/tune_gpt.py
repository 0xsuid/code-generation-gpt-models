import json
import torch
import psutil
import shutil
import requests
import pandas as pd
from pynvml import *
from os import makedirs
from datetime import date
from hashlib import sha256
# from urllib import parse
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi
from argparse import ArgumentParser
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, logging
import deepspeed
deepspeed.ops.adam.cpu_adam.CPUAdamBuilder().load()

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

def stop_tensordock_instance(api_key, api_token, server_id):
    req = requests.get(f"https://console.tensordock.com/api/stop/single?api_key={api_key}&api_token={api_token}&server={server_id}")
    return r.content

# Parse Arguments
parser = ArgumentParser()
parser.add_argument("-l", "--limit", dest="limit", default=0, type=int,
                    help="Limit Total no. of problems", metavar="N")
parser.add_argument("-upload", "--upload-model", dest="upload_model", action="store_true",
                    help="Upload fine-tuned model to Huggingface")
parser.add_argument("-stop", "--stop-instance", dest="stop_instance", action="store_true",
                    help="Stop tensordock instance after training")
parser.add_argument("-lr", "--local_rank", dest="local_rank",
                    help="local rank")
parser.add_argument("-ds", "--deepspeed", dest="deepspeed", default=None, type=str,
                    help="deepspeed config")
parser.add_argument("-v", "--verbosity", dest="verbosity", default="info", 
                    choices=["info","error"],
                    help="Verbosity", metavar="V")
args = parser.parse_args()

# load environment variables from .env
load_dotenv()  
td_api_key = os.getenv("TD_API_KEY")
td_api_token = os.getenv("TD_API_TOKEN")
td_server_id = os.getenv("TD_SERVER_ID")
huggingface_token = os.getenv("HF_TOKEN")
huggingface_repo_id = os.getenv("HF_REPO_ID")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", 
                                            bos_token='<|startoftext|>',
                                            eos_token='<|endoftext|>', 
                                            pad_token='<|pad|>')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").cuda()
model.resize_token_embeddings(len(tokenizer))

raw_ds = load_dataset("codeparrot/apps", split="train")
if(args.limit > 0):
    raw_ds = [x for _, x in zip(range(args.limit), raw_ds)]
coding_problems = format_input(raw_ds)

# max_length = max([len(tokenizer.encode(coding_problem, verbose=False)) for coding_problem in coding_problems])
model_max_length = model.config.max_position_embeddings
# Reset max_length to maximum model length if it exceeds. 
# max_length = max_length if max_length <= model_max_length else model_max_length
max_length = model_max_length
print("Max length: {}".format(max_length))


class AppsDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.coding_problems = txt_list
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        # for txt in txt_list:
        #     # truncation is required to avoid following issue
        #     # https://github.com/huggingface/transformers/issues/1791
        #     encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', 
        #                                 truncation=True,
        #                                 max_length=max_length, 
        #                                 padding="max_length")
        #     self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
        #     self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.coding_problems)

    def __getitem__(self, idx):
        # truncation is required to avoid following issue
        # https://github.com/huggingface/transformers/issues/1791
        encodings_dict = tokenizer('<|startoftext|>' + self.coding_problems[idx] + '<|endoftext|>', 
                                    truncation=True,
                                    max_length=max_length, 
                                    padding="max_length")
        return {
            "input_ids" : torch.tensor(encodings_dict['input_ids']),
            "attention_mask": torch.tensor(encodings_dict['attention_mask']),
            "labels" :  torch.tensor(encodings_dict['input_ids'])
        }


train_dataset = AppsDataset(coding_problems, tokenizer, max_length=max_length)
save_dir = './results'

# Logging - https://huggingface.co/docs/transformers/main_classes/logging
if(args.verbosity == "info"):
    logging.set_verbosity_info()
elif(args.verbosity == "error"):
    logging.set_verbosity_error()


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
    
    # Default "adamw_hf" is deprecated
    "optim": "adamw_torch",
    
    # The idea behind gradient accumulation is to instead of calculating the gradients for the whole batch at once to do it in smaller steps. 
    # The way we do that is to calculate the gradients iteratively in smaller batches by doing a forward and backward pass through the model and accumulating the gradients in the process. 
    # When enough gradients are accumulated we run the model’s optimization step. This way we can easily increase the overall batch size to numbers that would never fit into the GPU’s memory. 
    # In turn, however, the added forward and backward passes can slow down the training a bit.
    "gradient_accumulation_steps": 4,
    
    # In order to compute the gradients during the backward pass all activations from the forward pass are normally saved. 
    # This can create a big memory overhead. Alternatively, one could forget all activations during the forward pass and recompute them on demand during the backward pass. 
    # This would however add a significant computational overhead and slow down training.
    # "gradient_checkpointing":True,
    
    # Drop the last incomplete batch if it is not divisible by the batch size
    "dataloader_drop_last": True,
    
    # Number of steps used for a linear warmup from 0 to learning_rate. Overrides any effect of warmup_ratio.
    "warmup_steps": 100, 
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.
    "weight_decay": 0.01, 
    # The initial learning rate for AdamW optimizer.
    "learning_rate": 1e-5,
    
    # we can reduce the precision the variales and their computations are faster. 
    "fp16": True,
    "deepspeed": args.deepspeed
}
training_args = TrainingArguments(**default_args)
trainer = Trainer(model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        )

print_gpu_utilization()
result = trainer.train()
print_summary(result)

device_info = {
    "total_gpus": torch.cuda.device_count(),
    "v_cpus": psutil.cpu_count(),
    "total_memory_in_gb": psutil.virtual_memory().total/(1024*1024)
}
other_info= {
    "dataset_limit": args.limit,
}

all_configs = {**default_args,**device_info,**other_info}
configs_json = json.dumps(all_configs,sort_keys=True).encode('utf8')
calulated_hash = sha256(configs_json).hexdigest()
today = str(date.today())
final_save_dir = os.path.join("experiments", today+"-"+calulated_hash)

os.makedirs(final_save_dir,exist_ok=True)

with open(os.path.join(final_save_dir, 'configs.json'), 'w') as f:
    json.dump(all_configs, f, indent=4, ensure_ascii=False)

pwd_path = os.path.dirname(os.path.realpath(__file__))

# Move python stdout log "output.log" to final_save_dir
shutil.move(os.path.join(pwd_path, "output.log"), os.path.join(final_save_dir))

# Move Tensor logs to final_save_dir
shutil.move(os.path.join(pwd_path, "logs"), os.path.join(final_save_dir))

model.save_pretrained(os.path.join(final_save_dir, "final_checkpoint"))

if(args.upload_model 
   and huggingface_token
   and huggingface_repo_id):
    api = HfApi()
    api.upload_folder(
        folder_path=final_save_dir,
        path_in_repo=final_save_dir,
        repo_id=huggingface_repo_id,
        token=huggingface_token,
        # ignore_patterns="",
    )


if(args.stop_instance):
    if(td_api_key and td_api_token and td_server_id):
        stop_tensordock_instance(td_api_key, td_api_token, td_server_id)