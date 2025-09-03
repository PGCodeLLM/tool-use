time uv run generation_custom.py \
    --model "default" \
    --dataset "deathbyknowledge/shell-tasks" \
    --max-samples 2000 \
    --temperature 0.6 \
    --output-dir "qwen-8b-base" \
    --base-url "http://0.0.0.0:11153/v1" \
    --api-key "mykey233" \
    --sos-port 3002
