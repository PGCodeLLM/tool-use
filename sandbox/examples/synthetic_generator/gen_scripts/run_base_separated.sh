#!/bin/bash

# Optimized base model evaluation using separated phases
# This script uses generation_separated.py for better performance

time uv run generation_separated.py \
    --model "default" \
    --dataset "deathbyknowledge/shell-tasks" \
    --max-samples 2000 \
    --temperature 0.6 \
    --output-dir "qwen-8b-base-separated" \
    --base-url "http://0.0.0.0:11153/v1" \
    --api-key "mykey233" \
    --sos-port 3002 \
    --concurrency 6 \
    --pool-size 12 \
    --phase both