#!/bin/bash

# Phase 2 only: Execute pre-generated commands through sandboxes
# Requires commands.jsonl file from inference phase
# Uses optimized sandbox pooling and concurrency

# Default commands file location (edit as needed)
COMMANDS_FILE="test-base/commands.jsonl"

# Check if commands file exists
if [ ! -f "$COMMANDS_FILE" ]; then
    echo "ERROR: Commands file not found: $COMMANDS_FILE"
    echo "Please run inference phase first or specify correct path"
    echo ""
    echo "Usage: $0 [path-to-commands.jsonl]"
    exit 1
fi

# Use custom commands file if provided
if [ $# -gt 0 ]; then
    COMMANDS_FILE="$1"
fi

echo "=== Running Execution Phase Only ==="
echo "Using commands file: $COMMANDS_FILE"
echo "Concurrency: 8 workers, Pool size: 16 sandboxes"

time uv run generation_separated.py \
    --phase execution \
    --commands-file "$COMMANDS_FILE" \
    --output-dir "execution-only-output" \
    --sos-port 3000 \
    --concurrency 8 \
    --pool-size 16

echo "=== Execution Complete ==="
echo "Results saved to: execution-only-output/results.jsonl"