#!/bin/bash

# Optimized model evaluation with OpenHands (OH) using separated phases
# This script uses generation_separated.py for better performance

uv run generation_separated.py \
    --model "default" \
    --dataset "deathbyknowledge/shell-tasks" \
    --max-samples 2000 \
    --temperature 0.6 \
    --output-dir "qwen-8b-oh-separated" \
    --base-url "http://0.0.0.0:11152/v1" \
    --api-key "mykey233" \
    --sos-port 3001 \
    --concurrency 6 \
    --pool-size 12 \
    --phase both