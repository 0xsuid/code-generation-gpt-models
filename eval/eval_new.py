from evaluate import load
import json

apps_metric = load('codeparrot/apps_metric')
solution_sample1 = json.load(open("/root/code-generation-gpt-models/data/generated_code/GPT-Neo-1.3B/generated_codes_1_3B_100_random_problems_interview.json", "r"))
single_solutions = [[solution_sample1['0']['answer']]]

# print(solution_sample1['0']['answer'])

# to evaluate generations made for all levels for example
results = apps_metric.compute(predictions=single_solutions, level="interview")

print(results)