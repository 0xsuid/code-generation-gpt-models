#!/bin/bash

# Generate codes for GPT Neo 1.3B model

python3 generate_code.py \
        --limit 100 \
        --random \
        --difficulties "introductory" \
        --save "generated_codes_1_3B_100_random_problems_introductory.json" \
        -t "0xsuid/simba-1.3b" \
        -m "0xsuid/simba-1.3b"

python3 generate_code.py \
        --limit 100 \
        --random \
        --difficulties "interview" \
        --save "generated_codes_1_3B_100_random_problems_interview.json" \
        -t "0xsuid/simba-1.3b" \
        -m "0xsuid/simba-1.3b"

python3 generate_code.py \
        --limit 100 \
        --random \
        --difficulties "competition" \
        --save "generated_codes_1_3B_100_random_problems_competition.json" \
        -t "0xsuid/simba-1.3b" \
        -m "0xsuid/simba-1.3b"

# Generate codes for GPT Neo 125M model

# python3 generate_code.py \
#         --limit 100 \
#         --random \
#         --difficulties "introductory" \
#         --save "generated_codes_125M_100_random_problems_introductory.json" \
#         -t "EleutherAI/gpt-neo-125M" \
#         -m "0xsuid/simba-125M"

# python3 generate_code.py \
#         --limit 100 \
#         --random \
#         --difficulties "interview" \
#         --save "generated_codes_125M_100_random_problems_interview.json" \
#         -t "EleutherAI/gpt-neo-125M" \
#         -m "0xsuid/simba-125M"

# python3 generate_code.py \
#         --limit 100 \
#         --random \
#         --difficulties "competition" \
#         --save "generated_codes_125M_100_random_problems_competition.json" \
#         -t "EleutherAI/gpt-neo-125M" \
#         -m "0xsuid/simba-125M"