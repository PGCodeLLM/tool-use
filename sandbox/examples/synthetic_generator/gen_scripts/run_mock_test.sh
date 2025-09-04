#!/bin/bash

# Mock test script for pipeline validation
# Uses mock model (no API calls) for fast testing
# High concurrency to stress-test performance optimizations

echo "=== Mock Test - Pipeline Validation ==="
echo "This script tests the separated pipeline with mock model"
echo "No API calls will be made - useful for performance testing"

time uv run generation_separated.py \
    --model "mock" \
    --dataset "deathbyknowledge/shell-tasks" \
    --max-samples 50 \
    --temperature 0.6 \
    --output-dir "mock-test-results" \
    --sos-port 3000 \
    --concurrency 10 \
    --pool-size 20 \
    --phase both

echo "=== Mock Test Complete ==="
echo ""
echo "Files generated:"
echo "  - mock-test-results/commands.jsonl (generated commands)"
echo "  - mock-test-results/results.jsonl (execution results)"
echo "  - mock-test-results/summary.json (performance summary)"
echo ""
echo "This test validates:"
echo "  ✓ Sandbox pooling and reuse"
echo "  ✓ Concurrent execution"
echo "  ✓ Setup command grouping"
echo "  ✓ Pipeline separation"