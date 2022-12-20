import transformers
import json
import torch
from datasets import load_dataset

tokenizer = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = transformers.GPTNeoForCausalLM.from_pretrained("./experiments/2022-12-19-ab8f3a39c84fea7f66bf71860384bbce5df5fb3523e7dabd22b35c3ecfefb154/final_checkpoint")
model.resize_token_embeddings(len(tokenizer))

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

input_ids = torch.LongTensor(tokenizer.encode(coding_problems[0], verbose=True)).unsqueeze(0)
# input_ids = tokenizer(coding_problems[0], truncation=True, padding="max_length", return_tensors="pt")

output_ids = model.generate(
    input_ids,
    num_beams=5,
    early_stopping=True,
    max_length=2048 - len(input_ids)
)

print(tokenizer.decode(output_ids[0]))