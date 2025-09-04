#!/bin/bash

# Test the enhanced progress bar with a small mock dataset
# This will show both elapsed time and ETA columns

echo "=== Testing Enhanced Progress Bar ==="
echo "Running with mock model and 20 samples to see the new progress display"
echo ""

time uv run generation_separated.py \
    --model "mock" \
    --dataset "deathbyknowledge/shell-tasks" \
    --max-samples 20 \
    --temperature 0.6 \
    --output-dir "progress-bar-test" \
    --sos-port 3000 \
    --concurrency 4 \
    --pool-size 6 \
    --phase both

echo ""
echo "=== Test Complete ==="
echo "Check the progress bar display for:"
echo "  ✓ Spinner animation"
echo "  ✓ Phase labels (Phase 1: Generating commands, Phase 2: Executing commands)"
echo "  ✓ Progress bar visualization"
echo "  ✓ Task count (completed/total)"
echo "  ✓ Percentage complete"
echo "  ✓ Success rate (for execution phase)"
echo "  ✓ Elapsed time"
echo "  ✓ ETA (estimated time remaining)"