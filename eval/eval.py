import json
from evaluate import load

apps_metric = load('codeparrot/apps_metric')
generated_code = json.load(open("/home/user/code-generation-gpt-models/eval/generated_codes_1_3B_all_problems_introductory.json", "r"))
solutions = []

for problem in generated_code.values():
    solutions.append([problem['answer']])

# to evaluate generations made for all levels for example
results = apps_metric.compute(predictions=solutions, level="introductory")

print(results)