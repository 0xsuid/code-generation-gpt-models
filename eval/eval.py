import json
from evaluate import load
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="file", default='generated_codes.json',
                    help="Generated code file path")
parser.add_argument("-d", "--difficulty", dest="difficulty", choices=["all","introductory","interview","competition"],
                    default="all", help="difficulties - introductory, interview & competition")
args = parser.parse_args()

apps_metric = load('codeparrot/apps_metric')
generated_code = json.load(open(args.file, "r"))
solutions = []

for problem in generated_code.values():
    solutions.append([problem['answer']])

# to evaluate generations made for all levels for example
results = apps_metric.compute(predictions=solutions, level=args.difficulty)

print(results)