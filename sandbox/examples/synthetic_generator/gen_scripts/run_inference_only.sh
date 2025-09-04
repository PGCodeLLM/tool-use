#!/bin/bash

# Phase 1 only: Generate commands using the model (no sandbox execution)
# Useful for quick iteration on prompts and model settings
# Output: commands.jsonl file for later execution

echo "=== Running Inference Phase Only ==="
echo "This will generate commands without executing them in sandboxes"

uv run generation_separated.py \
    --model "default" \
    --dataset "deathbyknowledge/shell-tasks" \
    --max-samples 50 \
    --temperature 0.6 \
    --output-dir "test-base" \
    --base-url "http://0.0.0.0:11151/v1" \
    --api-key "mykey233" \
    --phase inference

echo "=== Inference Complete ==="
echo "Commands saved to: inference-only-output/commands.jsonl"
echo "To execute these commands, use run_execution_only.sh"