import transformers
import json
import torch
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = transformers.AutoTokenizer.from_pretrained("0xsuid/simba-1.3b")
model = transformers.AutoModelForCausalLM.from_pretrained("0xsuid/simba-1.3b")
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
generated_codes = {}

for idx, coding_problem in enumerate(coding_problems):
    encoded_tokens = tokenizer.encode(coding_problem)
    input_ids = torch.LongTensor(encoded_tokens).unsqueeze(0).to(device)

    output_ids = model.generate(
        input_ids,
        # num_beams=5,
        early_stopping=True,
        penalty_alpha=0.6, 
        top_k=4, 
        max_new_tokens=2048 - len(input_ids[0]),
        # max_length=2048 - len(input_ids[0])
    )

    generated_codes[idx] = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("ANSWER:\n")[1]

with open('generated_codes.json', "w") as f:
    json.dump(generated_codes, f)