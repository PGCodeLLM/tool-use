#!/bin/bash

# Optimized base model evaluation using separated phases
# This script uses generation_separated.py for better performance

time uv run generation_separated.py \
    --model "default" \
    --dataset "deathbyknowledge/shell-tasks" \
    --max-samples 2000 \
    --temperature 0.7 \
    --output-dir "results/qwen-8b-amir-direct0" \
    --base-url "http://0.0.0.0:11155/v1" \
    --api-key "mykey233" \
    --sos-port 3000 \
    --concurrency 10 \
    --batch-size 10 \
    --phase execution \
    --max-tokens 32000 \
    --commands-file results/qwen-8b-amir-direct0/commands.jsonl