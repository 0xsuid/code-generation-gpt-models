import transformers
import json
import torch
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = transformers.GPTNeoForCausalLM.from_pretrained("./experiments/neo-1_3b/experiments/2023-02-01-19cc6c3b87bd0a2fac9f1491ad836e1a7f445c9313854786ed97e8faa032593c/final_checkpoint")
model.resize_token_embeddings(len(tokenizer))
model.to(device)

def format_input(dataset):
        formatted_dataset = []
        for idx, data in enumerate(dataset):
                answer_type = "\nUse Call-Based format\n" if len(data["starter_code"])>0 else "\nUse Standard Input format\n"
                str_format = "\nQUESTION:\n" + data['question'] + "\n" + data["starter_code"] + "\n" + answer_type + "\nANSWER:\n"
                formatted_dataset.append(str_format)
        return formatted_dataset
    
raw_ds = load_dataset("codeparrot/apps", split="train")
coding_problems = format_input(raw_ds)

# print(coding_problems[0])

# coding_problems =  "\nQUESTION:\n" + "Get sum of two given input numbers x & y"  + "\n" + "\nUse Call-Based format\n" + "\nANSWER:\n"

input_ids = torch.LongTensor(tokenizer.encode(coding_problems[0], verbose=True)).unsqueeze(0).to(device)
# input_ids = tokenizer(coding_problems[0], truncation=True, padding="max_length", return_tensors="pt")

output_ids = model.generate(
    input_ids,
#     num_beams=5,
    early_stopping=True,
    penalty_alpha=0.6, 
    top_k=4, 
    max_new_tokens=120,
#     max_length=2048 - len(input_ids)
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))