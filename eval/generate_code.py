import transformers
import json
import time
import torch
import random
from argparse import ArgumentParser
from datasets import load_dataset


parser = ArgumentParser()
parser.add_argument("-l", "--limit", dest="limit", default=0, type=int,
                    help="Limit Total no. of problems")
parser.add_argument("-d", "--difficulties", dest="difficulties", choices=["all","introductory","interview","competition"],
                    default="all", help="difficulties - introductory, interview & competition")
parser.add_argument("-r", "--random", dest="random", action="store_true",
                    help="Randomize problem selection")
parser.add_argument("-dbg", "--debug", dest="debug", action="store_true",
                    help="Debug - print Generated Question & Answer")
parser.add_argument("-single", "--single", dest="single", default=None, type=int,
                    help="generate code for given problem id")
parser.add_argument("-s", "--save", dest="save", default="generated_codes.json",
                    help="Save Generated code to file")
parser.add_argument("-t", "--tokenizer", dest="tokenizer", default="0xsuid/simba-1.3b",
                    help="Tokenizer to use for code generation")
parser.add_argument("-m", "--model", dest="model", default="0xsuid/simba-1.3b",
                    help="Model to use for code generation")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer.pad_token = tokenizer.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

def format_input(dataset):
        formatted_dataset = []
        for idx, data in enumerate(dataset):
                answer_type = "\nUse Call-Based format\n" if len(data["starter_code"])>0 else "\nUse Standard Input format\n"
                str_format = "\nQUESTION:\n" + data['question'] + "\n" + data["starter_code"] + "\n" + answer_type + "\nANSWER:\n"
                formatted_dataset.append({"problem_id": data["problem_id"], "question": str_format, "input_output": data["input_output"]})
        return formatted_dataset

# No. of problems in each difficulty set
# introductory - 1000
# interview - 3000
# competition - 1000
raw_ds = load_dataset("codeparrot/apps", split="test", difficulties=[args.difficulties])
coding_problems = format_input(raw_ds)
generated_codes = {}

if(args.limit and args.limit>0):
    if(args.limit>0 and args.random):
        coding_problems = random.sample(coding_problems, args.limit)
    else:
        # Randomly Select given number(limit) of coding_problems
        coding_problems = coding_problems[:args.limit]

if(args.single):
        coding_problems = [coding_problems[args.single]]

print("Total Coding Problems: ",len(raw_ds))

for idx, coding_problem in enumerate(coding_problems):

    print("generating code for problem",idx)
    start = time.time()
    encoded_tokens = tokenizer.encode(coding_problem["question"])
    input_ids = torch.LongTensor(encoded_tokens).unsqueeze(0).to(device)
    print("Input problem token length: ",len(input_ids[0]))
    
    # Some Questions are too large than 1200 token so it doesn't make sense to trim question
    # and generate answer for that, so we will avoid those question and grab one of those extra problems
    if(len(input_ids[0]) > 1024):
        print("Token indices sequence length excceed than 1024: ",len(input_ids[0]))
        generated_codes[idx] = {"problem_id": coding_problem["problem_id"], "answer": ""}
        continue

    try:
        # https://arxiv.org/pdf/2202.06417.pdf - pg9
        # when α ∈ [0.5, 0.8], it yields generation diversity and
        # perplexity that are both comparable to human performance
        output_ids = model.generate(
            input_ids,
            
            # beam search
            num_beams=5,
            do_sample=True,
            early_stopping=True,
            max_new_tokens=1024,
            # no_repeat_ngram_size=2, 
            # num_return_sequences=5, 
            # max_length=2047
            
            # sample
            # do_sample=True, 
            # max_new_tokens=1024,
            # top_k=0, 
            # temperature=0.75,
            # early_stopping=True,
            
            # top-k
            # do_sample=True, 
            # early_stopping=True,
            # max_new_tokens=1024,
            # top_k=50,
            
            # top-p
            # do_sample=True, 
            # early_stopping=True,
            # max_new_tokens=1024,
            # top_p=0.8, 
            # top_k=0
            
            # top-p+top-k
            # do_sample=True, 
            # early_stopping=True,
            # max_new_tokens=1024,
            # top_k=50, 
            # top_p=0.93, 
            
            # Contrastive search
            # penalty_alpha=0.6,
            # top_k=4,
            # max_new_tokens=1024,
            # early_stopping=True,
            
        )
        generated_code = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("ANSWER:\n")[1]
        generated_codes[idx] = {"problem_id": coding_problem["problem_id"], "answer": generated_code}
        end = time.time()
        print("Time spent to Generate Code:", end - start)
        if(args.debug):
            print(coding_problem['question'])
            print(generated_code)
    except Exception as e:
        print("Failed To generate Code")
        print(e)
        break

with open(args.save, "w") as f:
    json.dump(generated_codes, f)