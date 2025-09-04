uv run generation_custom.py \
    --model "default" \
    --dataset "deathbyknowledge/shell-tasks" \
    --max-samples 2000 \
    --output-dir "qwen-8b-no-oh" \
    --base-url "http://0.0.0.0:11151/v1" \
    --api-key "mykey233" \
    --sos-port 3000

